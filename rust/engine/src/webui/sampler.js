    // --- Sampling Parameters Helper ---
    function getSampler() {
      return {
        temperature: parseFloat($('numTemp').value) || 0.7,
        top_p: parseFloat($('numTopP').value) || 0.95,
        top_k: parseInt($('numTopK').value, 10) || 0,
        min_p: parseFloat($('numMinP').value) || 0.0,
        max_tokens: parseInt($('numMaxTokens').value, 10) || 256,
        repetition_penalty: parseFloat($('numRepeatPen').value) || 1.0,
        presence_penalty: parseFloat($('numPresencePen').value) || 0.0,
        frequency_penalty: parseFloat($('numFreqPen').value) || 0.0,
        dry_multiplier: parseFloat($('numDryMult').value) || 0.0,
        mirostat: parseInt($('selMirostat').value, 10) || 0,
        seed: parseInt($('numSeed').value, 10),
        constraint: $('selConstraint').value,
        constraint_spec: $('txtConstraintSpec').value.trim(),
        speculative: $('chkSpeculative').checked,
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
      if (cfg.speculative !== undefined) $('chkSpeculative').checked = Boolean(cfg.speculative);
      if (cfg.system_prompt !== undefined && !$('txtSystemPrompt').value) {
        $('txtSystemPrompt').value = cfg.system_prompt;
      }
    }

    // Bind sliders <-> number inputs
    [['Temp', 'Temp'], ['TopP', 'TopP'], ['TopK', 'TopK'], ['MinP', 'MinP'], ['MaxTokens', 'MaxTokens'], ['RepeatPen', 'RepeatPen']].forEach(([n, r]) => {
      $(`rng${r}`).addEventListener('input', () => { $(`num${n}`).value = $(`rng${r}`).value; });
      $(`num${n}`).addEventListener('input', () => { $(`rng${r}`).value = $(`num${n}`).value; });
    });

