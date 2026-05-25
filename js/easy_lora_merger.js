import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

console.log("[EasyLoRAMerger] JS module loaded — registering extensions", Date.now());

// Enhanced method descriptions
const METHOD_DESCRIPTIONS = {
    "linear": "📊 Simple weighted average. Best for blending two similar styles. Use when you want a balanced mix.",
    "ties_strict": "⚔️ Resolves conflicts by keeping only where signs agree. Best for merging very different models where styles conflict.",
    "ties_gentle": "🕊️ A softer version of TIES. Keeps more data while reducing noise. Good for fine adjustments.",
    "slerp": "🌐 Spherical linear interpolation. Smooth directional blending between two vectors. Best for gradual transitions.",
    "subtract": "➖ Removes features of a model from another. Great for 'cleaning' unwanted styles or concepts.",
    "magnitude": "📈 Picks the strongest features from each. Good for high-detail merges where clarity matters.",
    "feature_mix": "🧬 Structural merge. Keeps unique architecture from both. Best for combining different styles.",
    "svd_preserve": "🔬 Mathematical reduction using SVD. Great for preserving character likeness while reducing size.",
    "noise_aware": "🧹 Filters out small weight fluctuations before merging. Good for cleaning noisy models.",
    "gradient_alignment": "🧭 Weights features based on directional alignment. Useful for refining similar styles."
};

// Ordered emoji label rules — first match wins.
// Place more specific rules (e.g. exact name matches) before broader prefix rules.
const WIDGET_LABEL_RULES = [
    // ── Core parameters (always green) ──
    { match: n => ["weight_a", "weight_b", "weight_c", "density", "balancing_mode",
                    "magnitude_scaling", "strength", "baking_method",
                    "energy_preservation", "blend_mode"].includes(n),
      emoji: "🟢" },

    // ── Save options ──
    { match: n => ["save_trigger", "save_folder", "filename"].includes(n),
      emoji: "💾" },

    // ── Performance options ──
    { match: n => ["device", "precision", "batch_size", "streaming"].includes(n),
      emoji: "⚙️" },

    // ── Metadata ──
    { match: n => ["metadata_mode", "keep_metadata"].includes(n),
      emoji: "📋" },

    // ── Component scaling (Weight Block Map) ──
    { match: n => ["weight_unet", "weight_clip", "weight_vae", "weight_te",
                    "te_mode", "te_weight"].includes(n),
      emoji: "🧱" },

    // ── SVD / Extraction parameters ──
    { match: n => ["rank", "alpha", "svd_mode", "energy_threshold",
                    "svd_energy_threshold"].includes(n),
      emoji: "🔬" },

    // ── Detection / Strength multipliers ──
    { match: n => ["detection_mode", "strength_multiplier"].includes(n),
      emoji: "🎯" },

    // ── Preview strength ──
    { match: n => ["strength_model", "strength_clip"].includes(n),
      emoji: "🎛️" },

    // ── Conversion settings (LoRA Studio) ──
    { match: n => ["target_format", "compression_mode", "target_rank",
                    "bake_custom_scale"].includes(n),
      emoji: "🔄" },

    // ── Strip settings (Checkpoint Studio) ──
    { match: n => ["strip_vae", "strip_te", "strip_clip"].includes(n),
      emoji: "✂️" },

    // ── Data passthrough inputs (place before generic prefix rules) ──
    { match: n => ["checkpoint_data", "lora_data"].includes(n),
      emoji: "📦" },

    // ── Compatibility mode (Checkpoint Studio) ──
    { match: n => n === "compatibility_mode",
      emoji: "🖥️" },

    // ── Input model / CLIP / LoRA / checkpoint selectors ──
    { match: n => ["model", "clip", "lora", "lora_name", "checkpoint"].includes(n),
      emoji: "📥" },

    // ── LoRA inputs (lora_a, lora_b, lora_c, etc.) ──
    { match: n => n.startsWith("lora_"),
      emoji: "📥" },

    // ── Checkpoint inputs (checkpoint_a, checkpoint_b, etc.) ──
    { match: n => n.startsWith("checkpoint_"),
      emoji: "📥" },
];

app.registerExtension({
    name: "EasyLoRA.Merger.SmartLabels",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const targetNodes = [
            "EasyLoRATripleMerger",
            "EasyCheckpointMerger",
            "EasyLoRAExtractor",
            "MusubiLoraConverter",
            "MusubiCheckpointStudio",
            "SmartModelBaker"
        ];

        if (targetNodes.includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // ── Update method description (only for nodes with a "method" widget) ──
                const updateMethodDescription = () => {
                    try {
                        const methodWidget = this.widgets.find(w => w.name === "method");
                        if (!methodWidget) return;

                        const method = methodWidget.value;

                        // Find or create the description widget
                        let descWidget = this.widgets.find(w => w.name === "method_description");
                        if (!descWidget) {
                            descWidget = this.addWidget("text", "method_description", "", () => {});
                            descWidget.name = "method_description";
                            descWidget.disabled = true;
                        }

                        // Update description
                        descWidget.value = METHOD_DESCRIPTIONS[method] || "Select a method to see details...";

                        // Style the info box to look like a label (guard against missing DOM element)
                        if (descWidget.inputEl) {
                            descWidget.inputEl.style.color = "#bbbbbb";
                            descWidget.inputEl.style.fontSize = "11px";
                            descWidget.inputEl.style.fontStyle = "italic";
                            descWidget.inputEl.style.lineHeight = "1.4";
                            descWidget.inputEl.style.border = "none";
                            descWidget.inputEl.style.background = "transparent";
                            descWidget.inputEl.style.padding = "4px 0px";
                            descWidget.inputEl.style.height = "45px";
                        }
                        descWidget.computedHeight = 45;
                    } catch (err) {
                        console.error("[EasyLoRA.SmartLabels] method description error:", err);
                    }
                };

                // ── Apply emoji-prefix labels to all widgets ──
                const updateWidgetLabels = () => {
                    try {
                        const methodWidget = this.widgets.find(w => w.name === "method");
                        const method = methodWidget ? methodWidget.value : null;

                        this.widgets.forEach(w => {
                            // Skip internal / control widgets
                            if (w.name === "method" || w.name === "method_description") return;

                            const name = w.name;

                            // ── Method-conditional highlighting ──
                            if (method) {
                                if (name === "uniqueness") {
                                    w.label = (method === "feature_mix") ? "🟢 " + name : "◌ " + name;
                                    return;
                                }
                                if (name === "threshold") {
                                    w.label = (method === "subtract") ? "🟢 " + name : "◌ " + name;
                                    return;
                                }
                                if (name === "blend") {
                                    w.label = (method === "magnitude") ? "🟢 " + name : "◌ " + name;
                                    return;
                                }
                            } else {
                                // No method widget — treat these as generic core params
                                if (["uniqueness", "threshold", "blend"].includes(name)) {
                                    w.label = "🟢 " + name;
                                    return;
                                }
                            }

                            // ── Apply standard emoji rules from the lookup table ──
                            for (const rule of WIDGET_LABEL_RULES) {
                                if (rule.match(name)) {
                                    w.label = rule.emoji + " " + name;
                                    return;
                                }
                            }

                            // Fallback: leave widget label untouched
                        });

                        this.setDirtyCanvas(true);
                    } catch (err) {
                        console.error("[EasyLoRA.SmartLabels] widget labels error:", err);
                    }
                };

                // ── Combined update ──
                const updateUI = () => {
                    updateMethodDescription();
                    updateWidgetLabels();
                };

                // Set up method change callback (only if method widget exists)
                const methodWidget = this.widgets.find(w => w.name === "method");
                if (methodWidget) {
                    const originalCallback = methodWidget.callback;
                    methodWidget.callback = (...args) => {
                        const res = originalCallback?.apply(methodWidget, args);
                        updateUI();
                        return res;
                    };
                }

                // Initial update with small delay to ensure widgets are ready
                setTimeout(updateUI, 10);
            };
        }
    }
});

// ═══════════════════════════════════════════════════════════════════════
// EasyTextDisplay — renders STRING input as a visible multiline widget
//
// Pattern follows proven ShowText (comfyui-custom-scripts) implementation:
//   1. Determine if there's a hidden converted-widget; clear from there
//   2. Create read-only widgets named "text_N" via ComfyWidgets
//   3. On execution: onExecuted → populate(message.text)
//   4. On workflow load: configure → store values, onConfigure → restore
// ═══════════════════════════════════════════════════════════════════════
app.registerExtension({
	name: "EasyLoRA.TextDisplay",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "EasyTextDisplay") {
			console.log("[EasyLoRA.TextDisplay] Registering for EasyTextDisplay node");

			/**
			 * Populate the node with read-only display widgets
			 * from the given text array.
			 *
			 * @param {string[]} text — e.g. ["hello world"]
			 */
			function populate(text) {
				try {
					if (this.widgets) {
						// On older frontend versions there is a hidden converted-widget
						const isConvertedWidget = +!!this.inputs?.[0]?.widget;
						for (let i = isConvertedWidget; i < this.widgets.length; i++) {
							this.widgets[i].onRemove?.();
						}
						this.widgets.length = isConvertedWidget;
					}

					// Guard: if text is undefined/null, there's nothing to display
					if (text == null) {
						console.warn("[EasyLoRA.TextDisplay] populate called with null/undefined text");
						return;
					}

					const v = [...text];
					if (!v[0]) {
						v.shift();
					}
					for (let list of v) {
						// Force list to be an array, not sure why sometimes it is/isn't
						if (!(list instanceof Array)) list = [list];
						for (const l of list) {
							const w = ComfyWidgets["STRING"](
								this,
								"text_" + (this.widgets?.length ?? 0),
								["STRING", { multiline: true }],
								app
							).widget;
							w.inputEl.readOnly = true;
							w.inputEl.style.opacity = 0.6;
							w.value = l;
						}
					}

					// Auto-resize to fit content
					requestAnimationFrame(() => {
						const sz = this.computeSize();
						if (sz[0] < this.size[0]) sz[0] = this.size[0];
						if (sz[1] < this.size[1]) sz[1] = this.size[1];
						this.onResize?.(sz);
						app.graph.setDirtyCanvas(true, false);
					});
				} catch (err) {
					console.error("[EasyLoRA.TextDisplay] populate() crashed:", err);
				}
			}

			// ── onExecuted: triggered when the node finishes running ──
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				try {
					onExecuted?.apply(this, arguments);
					console.log("[EasyLoRA.TextDisplay] onExecuted received:", JSON.stringify(message));
					populate.call(this, message?.text);
				} catch (err) {
					console.error("[EasyLoRA.TextDisplay] onExecuted crashed:", err);
				}
			};

			// ── Persist text across workflow saves/loads ──
			const VALUES = Symbol("textDisplayValues");
			const configure = nodeType.prototype.configure;
			nodeType.prototype.configure = function () {
				// Store unmodified widget values as they get removed on configure by new frontend
				this[VALUES] = arguments[0]?.widgets_values;
				return configure?.apply(this, arguments);
			};

			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function () {
				try {
					onConfigure?.apply(this, arguments);
					const widgets_values = this[VALUES];
					if (widgets_values?.length) {
						console.log("[EasyLoRA.TextDisplay] onConfigure restoring:", JSON.stringify(widgets_values));
						// In newer frontend there seems to be a delay in creating the initial widget
						requestAnimationFrame(() => {
							try {
								populate.call(
									this,
									widgets_values.slice(
										+(widgets_values.length > 1 && this.inputs?.[0]?.widget)
									)
								);
							} catch (err) {
								console.error("[EasyLoRA.TextDisplay] onConfigure populate crashed:", err);
							}
						});
					}
				} catch (err) {
					console.error("[EasyLoRA.TextDisplay] onConfigure crashed:", err);
				}
			};
		}
	},
});
