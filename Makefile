.PHONY: build build-wsclean build-r2d2 build-meqtrees build-polychord \
        smoke-test smoke-test-wsclean smoke-test-r2d2 nested-sampling-poc \
        shell-wsclean shell-r2d2 shell-meqtrees shell-polychord \
        fetch-r2d2-checkpoints record-environment plot-fits benchmark-report \
        config clean disk-usage

SHELL := /usr/bin/env bash

build: build-wsclean build-r2d2 build-meqtrees build-polychord

build-wsclean:
	scripts/build.sh wsclean

build-r2d2:
	scripts/build.sh r2d2

build-meqtrees:
	scripts/build.sh meqtrees

build-polychord:
	scripts/build.sh polychord

smoke-test: smoke-test-wsclean smoke-test-r2d2

smoke-test-wsclean:
	scripts/smoke-test-wsclean.sh

smoke-test-r2d2:
	scripts/smoke-test-r2d2.sh

nested-sampling-poc: build-wsclean build-meqtrees build-polychord
	scripts/run-nested-sampling-poc.sh

shell-wsclean:
	docker run --rm -it --platform linux/arm64 \
		-v "$$(pwd)/data:/data" -v "$$(pwd)/results:/results" \
		--entrypoint bash ri-reproducibility/wsclean:v3.7

shell-r2d2:
	docker run --rm -it --platform linux/arm64 \
		-v "$$(pwd)/data:/data" -v "$$(pwd)/checkpoints:/checkpoints" -v "$$(pwd)/results:/results" \
		--entrypoint bash ri-reproducibility/r2d2:cpu

shell-meqtrees:
	docker run --rm -it --platform linux/arm64 \
		-v "$$(pwd)/data:/data" -v "$$(pwd)/results:/results" \
		--entrypoint bash ri-reproducibility/meqtrees:kern-10

shell-polychord:
	docker run --rm -it --platform linux/arm64 \
		-v "$$(pwd):$$(pwd)" -w "$$(pwd)" \
		-v /var/run/docker.sock:/var/run/docker.sock \
		--entrypoint bash ri-reproducibility/polychord:lite

fetch-r2d2-checkpoints:
	scripts/fetch-r2d2-checkpoints.sh $(REALISATION)

record-environment:
	scripts/record-environment.sh --tool $(TOOL) --image $(IMAGE) $(if $(CONFIG),--config $(CONFIG),)

plot-fits:
	scripts/plot-fits.sh $(FILES)

benchmark-report:
	scripts/generate-benchmark-report.sh

config:
	docker compose config

disk-usage:
	docker system df -v

clean:
	-docker rmi ri-reproducibility/wsclean:v3.7 ri-reproducibility/wsclean:native ri-reproducibility/r2d2:cpu ri-reproducibility/meqtrees:kern-10 ri-reproducibility/polychord:lite 2>/dev/null
	rm -rf results/.smoke-test-fixtures results/smoke-test-wsclean results/smoke-test-r2d2
	@echo "Images and generated smoke-test outputs removed."
	@echo "data/, checkpoints/, and results/ contents you added yourself are left untouched."
	@echo "Run 'docker builder prune' to reclaim build-cache disk space if needed."
