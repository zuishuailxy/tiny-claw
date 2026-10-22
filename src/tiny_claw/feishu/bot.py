"""飞书机器人 — 长连接模式，无需 ngrok"""

import asyncio
import json
import logging
from typing import Any

from collections.abc import Callable

from lark_oapi import Client
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as WsClient
from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

from tiny_claw import config
from tiny_claw.engine import AgentEngine, Reporter
from tiny_claw.engine.session import Session, global_session_mgr
from tiny_claw.schema import Message, Role

logger = logging.getLogger("tiny-claw.feishu")

# 引擎工厂签名：接收 Session，返回装配完毕的 AgentEngine
EngineFactory = Callable[[Session], AgentEngine]


class FeishuReporter(Reporter):
    """将引擎输出格式化后发送到飞书"""

    def __init__(self, client: Client, chat_id: str):
        self._client = client
        self._chat_id = chat_id

    async def on_thinking(self, content: str) -> None:
        await self._send("🤔 模型正在慢思考 (Thinking)...")

    async def on_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        await self._send(
            f"🛠️ **正在执行工具**：`{tool_name}`\n"
            f"参数：`{json.dumps(args, ensure_ascii=False)}`"
        )

    async def on_tool_result(self, tool_name: str, output: str, is_error: bool) -> None:
        if is_error:
            await self._send(f"⚠️ **执行报错** ({tool_name})：\n{output}")
        else:
            await self._send(f"✅ **执行成功** ({tool_name})")

    async def on_message(self, content: str) -> None:
        await self._send(content)

    async def send_msg(self, text: str) -> None:
        """公开消息发送接口（供审批等外部模块使用）"""
        await self._send(text)

    async def _send(self, text: str) -> None:
        content_str = json.dumps({"text": text}, ensure_ascii=False)
        req = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(self._chat_id)
                .msg_type("text")
                .content(content_str)
                .build()
            )
            .build()
        )
        try:
            await self._client.im.v1.message.acreate(req)
        except Exception as e:
            logger.error("飞书消息发送失败: %s", e)


class FeishuBot:
    """飞书机器人 — 长连接模式 + 引擎工厂"""

    def __init__(self, factory: EngineFactory, work_dir: str):
        app_id = config.feishu_app_id()
        app_secret = config.feishu_app_secret()
        if not app_id or not app_secret:
            raise ValueError(
                "未找到飞书凭据：请在项目根 .env 中设置 "
                "FEISHU_APP_ID 和 FEISHU_APP_SECRET"
            )
        self._app_id = app_id
        self._app_secret = app_secret
        self._client = Client.builder().app_id(app_id).app_secret(app_secret).build()
        self._factory = factory  # 引擎工厂：每次收到消息时按 Session 动态创建
        self._work_dir = work_dir
        self._reporter: FeishuReporter | None = None
        logger.info("飞书机器人初始化完成（长连接模式）")

    def start(self) -> None:
        """启动长连接，阻塞运行"""

        def handle_message(event: Any) -> None:
            try:
                inner = event.event
                content = json.loads(inner.message.content).get("text", "")
                chat_id = inner.message.chat_id
                logger.info("收到会话 %s 消息: %s", chat_id, content[:100])

                # 检查是否为审批回复
                if content.startswith("approve ") or content.startswith("reject "):
                    self._handle_approval(chat_id, content)
                    return

                asyncio.create_task(self._handle_agent(chat_id, content))
            except Exception as e:
                logger.error("消息处理失败: %s", e)

        handler = (
            EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(handle_message)
            .build()
        )

        logger.info("🚀 tiny-claw 飞书长连接已启动，等待消息...")
        WsClient(self._app_id, self._app_secret, event_handler=handler).start()

    async def _handle_agent(self, chat_id: str, prompt: str) -> None:
        # 1. 为当前会话创建专属 Reporter
        self._reporter = FeishuReporter(self._client, chat_id)

        # 2. 获取/创建物理隔离的 Session（按 chat_id 隔离）
        sess = await global_session_mgr.get_or_create(chat_id, self._work_dir)
        await sess.append(Message(role=Role.USER, content=prompt))

        # 3. 通过工厂模式，为当前 Session 生成一个装配完毕的引擎
        #    工厂内部会将 CostTracker 绑定到该 Session，确保计费隔离
        engine = self._factory(sess)

        # 4. 注入 Reporter 并执行
        old = engine.reporter
        engine.reporter = self._reporter
        try:
            await engine.run(sess)
        except Exception as e:
            await self._reporter.send_msg(f"❌ Agent 运行崩溃: {e}")
        finally:
            engine.reporter = old

    @property
    def reporter(self) -> FeishuReporter | None:
        """返回当前绑定的 Reporter（供审批中间件等外部模块访问）"""
        return self._reporter

    def _handle_approval(self, chat_id: str, content: str) -> None:
        """处理飞书审批回复（approve/reject <task_id>）"""
        from tiny_claw.feishu.approve import global_approval_mgr

        parts = content.strip().split()
        if len(parts) != 2:
            logger.warning("审批命令格式错误: %s", content)
            return

        action, task_id = parts[0].lower(), parts[1]
        allowed = action == "approve"

        ok = global_approval_mgr.resolve_approval(
            task_id,
            allowed=allowed,
            reason=f"用户从飞书 {'批准' if allowed else '拒绝'}",
        )
        if ok:
            # 异步发送确认消息
            reporter = FeishuReporter(self._client, chat_id)
            asyncio.create_task(
                reporter.send_msg(
                    f"{'✅ 已批准' if allowed else '❌ 已拒绝'}任务 `{task_id}`"
                )
            )
        else:
            logger.info("审批 TaskID %s 未找到或已完成", task_id)
