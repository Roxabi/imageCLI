SUPERVISOR_HUB ?= $(HOME)/projects
HUB_SERVICES   := gen comfyui
-include $(SUPERVISOR_HUB)/hub.mk

.PHONY: register gen comfyui install lint test

register:
	@echo "Registering imageCLI with supervisor hub..."
	@$(HUB_GEN_MK) imagecli "$(abspath .)" gen comfyui
	$(call hub-link-conf,imagecli_gen,supervisor/conf.d/imagecli_gen.conf)
	@mkdir -p "$(HOME)/.local/state/imagecli/logs"
	$(hub_reread)
	@echo "Done. Run 'make gen' to start the generation daemon."

gen:
	$(ensure_hub)
	@$(HUB_SVC) imagecli_gen $(SVC_CMD)

# ── ComfyUI (standalone, not managed by supervisor) ──────────────────────────
COMFYUI_DIR := $(HOME)/ComfyUI
COMFYUI_PID := /tmp/comfyui.pid
COMFYUI_LOG := /tmp/comfyui.log

comfyui:
	@case "$(SVC_CMD)" in \
		logs)   tail -f $(COMFYUI_LOG) ;; \
		stop)   kill $$(cat $(COMFYUI_PID) 2>/dev/null) 2>/dev/null && echo "ComfyUI stopped" || echo "ComfyUI not running"; rm -f $(COMFYUI_PID) ;; \
		status) pgrep -f "$(COMFYUI_DIR)/venv/bin/python" > /dev/null && echo "ComfyUI running" || echo "ComfyUI not running" ;; \
		*)      [ -d "$(COMFYUI_DIR)" ] || { echo "Error: ComfyUI not found at $(COMFYUI_DIR)"; exit 1; }; \
			pgrep -qf "$(COMFYUI_DIR)/venv/bin/python" && echo "ComfyUI already running. Use 'make comfyui stop'." && exit 0 || true; \
			echo "Starting ComfyUI at http://localhost:8188"; \
			cd $(COMFYUI_DIR) && nohup venv/bin/python main.py --listen 127.0.0.1 --port 8188 > $(COMFYUI_LOG) 2>&1 & echo $$! > $(COMFYUI_PID); \
			echo "Started — PID $$(cat $(COMFYUI_PID))" ;; \
	esac

install:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest
