RULE: CLAUDE DOES NOT WRITE HERE

1. Pipeline: headset mic -> link to MacBook -> faster-whisper, base.en -> text normalier -> MiniLM -> calibrated reject -> closed-list slot resolver -> result back over link -> HUD
2. Location: Whisper and MiniLM are on the MacBook and HUD reaches it by either cable or WiFi
3. Model: MiniLM -> fine-tuned + distilled from bigger teachers -> exported to ONNX INT8
4. Output: {text, selected_intent, confidence, top5, slot} + low confidence -> unahndled