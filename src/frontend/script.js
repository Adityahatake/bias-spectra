document.addEventListener('DOMContentLoaded', () => {
    const headlineEl = document.getElementById('headline');
    const modelEl = document.getElementById('model');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');

    const resultPanel = document.getElementById('result-panel');
    const emptyState = document.getElementById('empty-state');
    const resultContent = document.getElementById('result-content');
    
    const badgeEl = document.getElementById('gate-badge');
    const biasLabelEl = document.getElementById('bias-label');
    const biasReasoningEl = document.getElementById('bias-reasoning');
    
    const confSection = document.getElementById('confidence-section');
    const confTracksEl = document.getElementById('conf-tracks');
    
    const historyTrack = document.getElementById('history-track');
    let history = [];

    const COLORS = {
        'Left': 'var(--color-left)',
        'Neutral': 'var(--color-neutral)',
        'Right': 'var(--color-right)'
    };
    
    const BG_COLORS = {
        'Left': 'linear-gradient(90deg, #ef4444, #f87171)',
        'Neutral': 'linear-gradient(90deg, #eab308, #facc15)',
        'Right': 'linear-gradient(90deg, #3b82f6, #60a5fa)'
    };

    analyzeBtn.addEventListener('click', async () => {
        const headline = headlineEl.value.trim();
        const model = modelEl.value;

        if (!headline) {
            headlineEl.style.borderColor = 'var(--color-left)';
            setTimeout(() => headlineEl.style.borderColor = 'rgba(255, 255, 255, 0.1)', 1000);
            return;
        }

        // Set Loading State
        analyzeBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');

        try {
            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ headline, model })
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "API Error");
            }
            
            const data = await resp.json();
            renderResult(data);
            addToHistory(headline, data.label);

        } catch (error) {
            console.error("Error analyzing bias:", error);
            alert("Error: " + error.message);
        } finally {
            analyzeBtn.disabled = false;
            loader.classList.add('hidden');
            btnText.classList.remove('hidden');
        }
    });

    function renderResult(data) {
        emptyState.classList.add('hidden');
        resultContent.classList.remove('hidden');
        
        // Render Badge
        let badgeStyle = "background: rgba(102,126,234,0.15); color: #818cf8; border-color: rgba(102,126,234,0.3);";
        let gateText = `⚡ ML Model (${modelEl.value.toUpperCase()})`;
        
        if (data.gate === 'non_political') {
            badgeStyle = "background: rgba(16,185,129,0.15); color: #10b981; border-color: rgba(16,185,129,0.3);";
            gateText = "⚡ Non-Political Filter";
        } else if (data.gate === 'neutral_political') {
            badgeStyle = "background: rgba(234,179,8,0.15); color: #facc15; border-color: rgba(234,179,8,0.3);";
            gateText = "⚡ Neutral Filter";
        }
        badgeEl.style.cssText = badgeStyle;
        badgeEl.textContent = gateText;

        // Render Bias Outcome
        biasLabelEl.textContent = data.label;
        biasLabelEl.style.color = COLORS[data.label] || 'white';
        biasReasoningEl.textContent = data.reasoning;

        // Render Confidence Models
        confTracksEl.innerHTML = '';
        if (data.is_model_prediction && data.confidence) {
            confSection.classList.remove('hidden');
            
            ['Left', 'Neutral', 'Right'].forEach(lbl => {
                const conf = data.confidence[lbl] || 0;
                const pct = (conf * 100).toFixed(1);
                
                const row = document.createElement('div');
                row.className = 'conf-row';
                
                row.innerHTML = `
                    <div class="conf-header">
                        <span>${lbl}</span>
                        <span style="color: ${COLORS[lbl]}">${pct}%</span>
                    </div>
                    <div class="conf-track-bg">
                        <div class="conf-track-fill" style="width: 0%; background: ${BG_COLORS[lbl]}"></div>
                    </div>
                `;
                confTracksEl.appendChild(row);
                
                // Trigger animation
                setTimeout(() => {
                    const fill = row.querySelector('.conf-track-fill');
                    if(fill) {
                        fill.style.width = pct + '%';
                    }
                }, 50);
            });
        } else {
            confSection.classList.add('hidden');
        }
    }

    function addToHistory(headline, label) {
        // truncate headline
        const short = headline.length > 50 ? headline.substring(0, 50) + "..." : headline;
        
        // Maintain history list limit 10
        if(history.length >= 10) history.pop();
        history.unshift({ short, label });

        const empty = historyTrack.querySelector('.history-empty');
        if (empty) empty.remove();
        
        // Map label class
        const klass = label.toLowerCase();
        const symbol = klass === 'left' ? '🟥' : klass === 'right' ? '🟦' : '🟡';
        
        const span = document.createElement('div');
        span.className = `history-item ${klass}`;
        span.innerHTML = `<span class="dot">${symbol}</span> ${short}`;
        
        historyTrack.insertBefore(span, historyTrack.firstChild);
    }
});
