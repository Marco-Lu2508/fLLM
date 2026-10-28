    // --- Sampling Parameters Helper ---
    function getSampler() {
      return {
        temperature: parseFloat($('numTemp').value) || 0.7,
        top_p: parseFloat($('numTopP').value) || 0.95,
        top_k: parseInt($('numTopK').value, 10) || 0,
        min_p: parseFloat($('numMinP').value) || 0.0,
        max_tokens: parseInt($('numMaxTokens').value, 10) || 0, // 0: automatic, see requestBody
        repetition_penalty: parseFloat($('numRepeatPen').value) || 1.0,
        presence_penalty: parseFloat($('numPresencePen').value) || 0.0,
        frequency_penalty: parseFloat($('numFreqPen').value) || 0.0,
        dry_multiplier: parseFloat($('numDryMult').value) || 0.0,
        mirostat: parseInt($('selMirostat').value, 10) || 0,
        seed: parseInt($('numSeed').value, 10),
        constraint: $('selConstraint').value,
        constraint_spec: $('txtConstraintSpec').value.trim(),
        logprobs: $('chkLogprobs').checked,
        system_prompt: $('txtSystemPrompt').value.trim(),
        stop: $('inpStopWords').value.split(',').map((s) => s.trim()).filter(Boolean),
      };
    }

    function setSamplerUI(cfg) {
      if (!cfg) return;
      const pairs = [
        ['Temp', cfg.temperature],
        ['TopP', cfg.top_p],
        ['TopK', cfg.top_k],
        ['MinP', cfg.min_p],
        ['RepeatPen', cfg.repetition_penalty],
      ];
      pairs.forEach(([k, v]) => {
        if (v !== undefined && v !== null) {
          $(`num${k}`).value = v;
          $(`rng${k}`).value = v;
        }
      });
      if (cfg.presence_penalty !== undefined) $('numPresencePen').value = cfg.presence_penalty;
      if (cfg.frequency_penalty !== undefined) $('numFreqPen').value = cfg.frequency_penalty;
      if (cfg.dry_multiplier !== undefined) $('numDryMult').value = cfg.dry_multiplier;
      if (cfg.mirostat !== undefined) $('selMirostat').value = String(cfg.mirostat);
      if (cfg.seed !== undefined) $('numSeed').value = cfg.seed === null ? -1 : cfg.seed;
      if (cfg.system_prompt !== undefined && !$('txtSystemPrompt').value) {
        $('txtSystemPrompt').value = cfg.system_prompt;
      }
    }

    /** The range of the Max tokens slider follows the context window of the loaded model (0 on the slider is automatic). */
    function syncMaxTokensRange() {
      const ctx = Number(state.status?.model?.max_context) || 8192;
      const max = Math.max(8192, Math.min(ctx, 1048576));
      const range = $('rngMaxTokens');
      if (Number(range.max) !== max) { range.max = max; range.step = max > 65536 ? 1024 : max > 16384 ? 256 : 16; }
    }

    // Bind sliders <-> number inputs
    [['Temp', 'Temp'], ['TopP', 'TopP'], ['TopK', 'TopK'], ['MinP', 'MinP'], ['MaxTokens', 'MaxTokens'], ['RepeatPen', 'RepeatPen']].forEach(([n, r]) => {
      $(`rng${r}`).addEventListener('input', () => { $(`num${n}`).value = $(`rng${r}`).value; });
      $(`num${n}`).addEventListener('input', () => { $(`rng${r}`).value = $(`num${n}`).value; });
    // 0 on the Max tokens slider means automatic: the number field then stays empty
    $('rngMaxTokens').addEventListener('input', () => { if (Number($('rngMaxTokens').value) === 0) $('numMaxTokens').value = ''; });
    });


    // --- Thinking chip: what the loaded model can do, and the level chosen for this chat ---
    // @pure-begin
    const THINKING_LEVELS = { off: 'Off', low: 'Low', medium: 'Medium', high: 'High' };
    const THINKING_NO_CAPS = { supported: false, control: 'none', variable: '', levels: [], default: 'off' };

    /** What the server says about thinking, as the page uses it. `remote`: a model of another provider (nothing is known about it). */
    function thinkingCapsOf(raw, remote) {
      if (remote) return { supported: true, control: 'remote', variable: '', levels: ['off', 'low', 'medium', 'high'], default: 'off' };
      if (raw && typeof raw === 'object') {
        const control = ['none', 'toggle', 'effort', 'always'].includes(raw.control) ? raw.control : (raw.supported ? 'toggle' : 'none');
        const levels = (Array.isArray(raw.levels) ? raw.levels : []).filter((l) => Object.prototype.hasOwnProperty.call(THINKING_LEVELS, l));
        if (raw.supported === false || control === 'none' || !levels.length) return THINKING_NO_CAPS;
        return { supported: true, control, variable: String(raw.variable || ''), levels, default: levels.includes(raw.default) ? raw.default : levels[0] };
      }
      // an older server only said whether the template mentions thinking: a plain switch
      if (raw === true) return { supported: true, control: 'toggle', variable: 'enable_thinking', levels: ['off', 'high'], default: 'off' };
      return THINKING_NO_CAPS;
    }

    /** The level to use: the wanted one when the model has it, otherwise the model's own default. `null` without thinking. */
    function thinkingLevelFor(caps, wanted) {
      if (!caps.supported) return null;
      return caps.levels.includes(wanted) ? wanted : caps.default;
    }

    /** The next level when the chip is clicked (the last one wraps to the first). */
    function nextThinkingLevel(caps, level) {
      if (!caps.supported || caps.levels.length < 2) return level;
      return caps.levels[(caps.levels.indexOf(level) + 1) % caps.levels.length];
    }

    /** The fields of a chat request for the level. The server turns low/medium into a thinking budget on models with a switch. */
    function thinkingRequestFields(caps, level) {
      if (!caps.supported || !level) return {};
      if (caps.control === 'effort') return { reasoning_effort: level };
      if (caps.control === 'remote') return level === 'off' ? {} : { reasoning_effort: level };
      if (caps.control !== 'toggle') return {};
      const out = {};
      if (level !== 'off') out.reasoning_effort = level;
      if (caps.variable) out.chat_template_kwargs = { [caps.variable]: level !== 'off' };
      return out;
    }

    /** How the chip looks: `{visible, disabled, active, label, title}`. */
    function thinkingChipView(caps, level) {
      if (!caps.supported) return { visible: false, disabled: true, active: false, label: 'Thinking', title: 'This model has no thinking mode.' };
      if (caps.control === 'always') {
        return { visible: true, disabled: true, active: true, label: 'Thinking', title: 'This model always thinks before it answers; there is nothing to switch.' };
      }
      const name = THINKING_LEVELS[level] || '';
      const budgets = caps.control === 'toggle' ? ' Low and Medium stop the thinking after about 512 and 2048 tokens and let the model answer.' : '';
      const remote = caps.control === 'remote' ? ' Sent as the reasoning effort; models that cannot think may refuse it.' : '';
      return {
        visible: true,
        disabled: false,
        active: level !== 'off',
        label: level === 'off' || !name ? 'Thinking' : `Thinking: ${name}`,
        title: `Thinking: ${name || 'default'}. Click to change.${budgets}${remote}`,
      };
    }

    /** What the load dialog tells about a speculative decoding mode. */
    function speculativeNote(mode) {
      const what = {
        lookup: 'Guesses the next words from the text of the conversation and lets the model check them. It helps with code edits, summaries and quoted text; in free chat the guesses are often wrong and it can make the answer a little slower (measured: 17 % slower on a story). Conversation lookup is the better choice.',
        suffix: 'Looks for the same words earlier in the conversation and in recent answers and lets the model check the guess. It costs nothing when it finds no match and roughly doubles the speed when the model repeats text: editing code, summaries, quotes, agent loops that read the same files again. The answer is the same as without it.',
        draft: 'A small model of the same family guesses ahead and this model checks the guesses. It needs extra memory and helps with every kind of text when the small model often agrees.',
        mtp: 'Uses the prediction heads that some models carry. Only models that have them can use it, on the processor or a CUDA graphics card.',
      }[mode];
      if (!what) return mode === 'off' ? 'Every word is computed one at a time.' : '';
      return `${what} The answer is the same as without it (sampled answers keep their distribution); only the speed changes. Models with recurrent layers (for example Qwen3.5 or Mamba) cannot use it.`;
    }
    // @pure-end

    function thinkingCaps() {
      const remote = typeof parseRemoteModelId === 'function' && Boolean(parseRemoteModelId(state.selectedModel));
      return thinkingCapsOf(state.status?.model?.capabilities?.thinking, remote);
    }

    /** The level of the open chat (a new chat uses the last choice). */
    function thinkingLevelNow() {
      const t = typeof currentThread === 'function' ? currentThread() : null;
      return thinkingLevelFor(thinkingCaps(), (t && t.thinkingLevel) || state.thinkingLevel);
    }

    function setThinkingLevel(level) {
      state.thinkingLevel = level;
      const t = typeof currentThread === 'function' ? currentThread() : null;
      if (t) { t.thinkingLevel = level; saveThreads(); }
      renderThinkingChip();
    }

    function renderThinkingChip() {
      const chip = $('btnToggleThinking');
      if (!chip) return;
      const caps = thinkingCaps();
      const level = thinkingLevelNow();
      const view = thinkingChipView(caps, level);
      chip.style.display = view.visible ? '' : 'none';
      chip.disabled = view.disabled;
      chip.classList.toggle('active', view.active);
      chip.title = view.title;
      $('thinkingChipLabel').textContent = view.label;
      state.thinkingEnabled = Boolean(caps.supported && level && level !== 'off'); // the code page reads this flag
    }
