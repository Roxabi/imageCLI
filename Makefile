SUPERVISOR_HUB ?= $(HOME)/projects
HUB_SERVICES   := gen
-include $(SUPERVISOR_HUB)/hub.mk

.PHONY: register gen install lint test

register:
	@echo "Registering imageCLI with supervisor hub..."
	@$(HUB_GEN_MK) imagecli "$(abspath .)" gen
	$(call hub-link-conf,imagecli_gen,supervisor/conf.d/imagecli_gen.conf)
	@mkdir -p "$(HOME)/.local/state/imagecli/logs"
	$(hub_reread)
	@echo "Done. Run 'make gen' to start the generation daemon."

gen:
	$(ensure_hub)
	@$(HUB_SVC) imagecli_gen $(SVC_CMD)

install:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest
