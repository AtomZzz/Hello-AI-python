const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const statusEl = document.getElementById('status');

function renderMessage(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLoading(loading) {
    sendBtn.disabled = loading;
    statusEl.textContent = loading ? 'AI 思考中...' : '';
}

async function sendMessage() {
    const message = inputEl.value.trim();
    if (!message) {
        return;
    }

    renderMessage('user', message);
    inputEl.value = '';
    setLoading(true);

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.error || '请求失败');
        }

        renderMessage('ai', data.reply || '无回复');
        if (data.route) {
            statusEl.textContent = `[ROUTER] source=${data.route.source || 'unknown'} use_agent=${data.route.use_agent} use_rag=${data.route.use_rag} require_json=${data.route.require_json}`;
        }
    } catch (err) {
        renderMessage('ai', `请求异常: ${err.message}`);
        statusEl.textContent = '请求失败，请稍后重试。';
    } finally {
        sendBtn.disabled = false;
    }
}

sendBtn.addEventListener('click', sendMessage);

inputEl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

