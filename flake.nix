{
  description = "Cell2Image toolbox";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.pyproject-nix.url = "github:nix-community/pyproject.nix";
  inputs.pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";

  outputs = { self, nixpkgs, flake-utils, pyproject-nix }:
  flake-utils.lib.eachDefaultSystem
    (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
        # system_mpfm = import mpfm { inherit system; };
        pythonDevPkgs = with pkgs; [
          nodePackages.pyright
          python3Packages.pylsp-rope
          python3Packages.python-lsp-black
          python3Packages.python-lsp-server
          python3Packages.rope
        ];

        project = pyproject-nix.lib.project.loadPyproject {
          projectRoot = ./.;
        };
        python = pkgs.python3;
      in rec {

        # take the mpfm pyShellCuda dev shell
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            (pkgs.python3.withPackages (p: with p; [
              numpy
              matplotlib
              ipython
              click
              vtk
              pillow
              numba
              virtualenv
              ffmpeg-python
              ffmpy
              tqdm
            ]))
            neovim
          ] ++ pythonDevPkgs;
          shellHook = ''
            # Allow the use of wheels.
            SOURCE_DATE_EPOCH=$(date +%s)
            # Setup the virtual environment if it doesn't already exist.
            VENV=.venv
            if test ! -d $VENV; then
              virtualenv $VENV
            fi
            source ./$VENV/bin/activate
            export PYTHONPATH=`pwd`/$VENV/${pkgs.python3.sitePackages}/:$PYTHONPATH

            pip install -e .
          '';
        };

        packages.default =
          let
            # Returns an attribute set that can be passed to `buildPythonPackage`.
            attrs = project.renderers.buildPythonPackage { inherit python; };
          in
          # Pass attributes to buildPythonPackage.
            # Here is a good spot to add on any missing or custom attributes.
          python.pkgs.buildPythonPackage (attrs);
      }
    );
}
