import { app } from "../../scripts/app.js";

// Enhanced method descriptions
const METHOD_DESCRIPTIONS = {
    "linear": "📊 Simple weighted average. Best for blending two similar styles. Use when you want a balanced mix.",
    "ties_strict": "⚔️ Resolves conflicts by keeping only where signs agree. Best for merging very different LoRAs where styles conflict.",
    "ties_gentle": "🕊️ A softer version of TIES. Keeps more data while reducing noise. Good for fine adjustments.",
    "dare_lite": "🎲 Experimental: Randomly drops weights to keep the LoRA small. Can create interesting variations.",
    "dare_rescale": "⚖️ Drops weights but rescales others to maintain overall strength. Good for sparsification.",
    "subtract": "➖ Removes features of LoRA B from LoRA A. Great for 'cleaning' unwanted styles or concepts.",
    "magnitude": "📈 Picks the strongest features from each. Good for high-detail merges where clarity matters.",
    "feature_mix": "🧬 Structural merge. Keeps unique architecture from both. Best for combining different styles.",
    "svd_preserve": "🔬 Mathematical reduction using SVD. Great for preserving character likeness while reducing size.",
    "noise_aware": "🧹 Filters out small weight fluctuations before merging. Good for cleaning noisy LoRAs.",
    "gradient_alignment": "🧭 Weights features based on directional alignment. Useful for refining similar styles."
};

app.registerExtension({
    name: "EasyLoRA.Merger.SmartLabels",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const targetNodes = ["EasyLoRAmerger", "EasyLoRAonlyMerger", "EasyLoRATripleMerger"];
        
        if (targetNodes.includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const updateUI = () => {
                    const methodWidget = this.widgets.find(w => w.name === "method");
                    if (!methodWidget) return;
                    const method = methodWidget.value;
                    
                    // Find or create the description widget
                    let descWidget = this.widgets.find(w => w.name === "method_info");
                    if (!descWidget) {
                        descWidget = this.addWidget("text", "method_info", "", () => {});
                        descWidget.name = "method_info";
                        descWidget.disabled = true;
                    }
                    
                    // Update description
                    descWidget.value = METHOD_DESCRIPTIONS[method] || "Select a method to see details...";
                    // Style the info box to look like a label
                    descWidget.inputEl.style.color = "#bbbbbb";       
                    descWidget.inputEl.style.fontSize = "11px";
                    descWidget.inputEl.style.fontStyle = "italic";
                    descWidget.inputEl.style.lineHeight = "1.4";
                    descWidget.inputEl.style.border = "none";
                    descWidget.inputEl.style.background = "transparent";
                    descWidget.inputEl.style.padding = "4px 0px";
                    descWidget.computedHeight = 45;                   // Force the widget to be taller
                    descWidget.inputEl.style.height = "45px";
                    
                    // Update all labels
                    this.widgets.forEach(w => {
                        if (w.name === "method" || w.name === "method_info") return;
                        
                        const name = w.name;
                        
                        // Core parameters (always green)
                        if (["weight_a", "weight_b", "weight_c", "density"].includes(name)) {
                            w.label = "🟢 " + name;
                        } 
                        // Feature mix
                        else if (name === "uniqueness") {
                            w.label = (method === "feature_mix") ? "🟢 " + name : "◌ " + name;
                        }
                        // Subtract
                        else if (name === "threshold") {
                            w.label = (method === "subtract") ? "🟢 " + name : "◌ " + name;
                        }
                        // Magnitude
                        else if (name === "blend") {
                            w.label = (method === "magnitude") ? "🟢 " + name : "◌ " + name;
                        }
                        // Save options
                        else if (["save_trigger", "save_folder", "filename"].includes(name)) {
                            w.label = "💾 " + name;
                        }
                        // Performance options
                        else if (["device", "precision", "batch_size", "streaming"].includes(name)) {
                            w.label = "⚙️ " + name;
                        }
                        // Metadata
                        else if (name === "metadata_mode") {
                            w.label = "📋 " + name;
                        }
                        // LoRA inputs
                        else if (name.startsWith("lora_")) {
                            w.label = "📥 " + name;
                        }
                    });
                    
                    this.setDirtyCanvas(true);
                };

                // Set up method change callback
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