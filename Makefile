ifneq (,$(filter gen,$(firstword $(MAKECMDGOALS))))
  SVC_CMD := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  ifneq (,$(SVC_CMD))
    $(eval $(SVC_CMD):;@:)
  endif
endif

SUPERVISOR_HUB ?= $(HOME)/projects
SUPERVISORCTL  := $(SUPERVISOR_HUB)/scripts/supervisorctl.sh
SUPERVISOR_START := $(SUPERVISOR_HUB)/scripts/start.sh
HUB_PID        := $(SUPERVISOR_HUB)/supervisord.pid

define ensure_hub
	@if [ ! -f "$(HUB_PID)" ] || ! kill -0 $$(cat "$(HUB_PID)" 2>/dev/null) 2>/dev/null; then \
		echo "Hub supervisord not running, starting..."; \
		$(SUPERVISOR_START); \
	fi
endef

.PHONY: register gen install lint test

register:
	@echo "Registering imageCLI with supervisor hub..."
	@mkdir -p "$(SUPERVISOR_HUB)/conf.d"
	@ln -sf "$(abspath supervisor/conf.d/imagecli_gen.conf)" "$(SUPERVISOR_HUB)/conf.d/imagecli_gen.conf"
	@mkdir -p "$(HOME)/.local/state/imagecli/logs"
	@if [ -S "$(SUPERVISOR_HUB)/supervisor.sock" ]; then \
		$(SUPERVISORCTL) reread && $(SUPERVISORCTL) update; \
	fi
	@echo "Done. Run 'make gen' to start the generation daemon."

gen:
	$(ensure_hub)
ifeq ($(SVC_CMD),reload)
	@$(SUPERVISORCTL) restart imagecli_gen
else ifeq ($(SVC_CMD),logs)
	@$(SUPERVISORCTL) tail -f imagecli_gen
else ifeq ($(SVC_CMD),errlogs)
	@$(SUPERVISORCTL) tail -f imagecli_gen stderr
else ifeq ($(SVC_CMD),stop)
	@$(SUPERVISORCTL) stop imagecli_gen
else ifeq ($(SVC_CMD),start)
	@$(SUPERVISORCTL) start imagecli_gen
else ifeq ($(SVC_CMD),status)
	@$(SUPERVISORCTL) status imagecli_gen
else
	@$(SUPERVISORCTL) start imagecli_gen
endif

install:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest
