{ pkgs, lib, config, inputs, ... }:

{
  # Allow build tools in the environment
  env.LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

  # https://devenv.sh/basics/
  env.GREET = "lis2";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.gcc
    pkgs.pkg-config
    pkgs.python313Packages.numpy
    pkgs.python313Packages.scipy
    ];

  # https://devenv.sh/languages/
  # languages.rust.enable = true;
  languages.python = {
    enable = true;
    version = "3.13";
    venv.enable = true;
    uv = {
      enable = true;
      sync = {
        enable = true;
        groups = ["dev" "doc"];
      };
    };
  };

  # https://devenv.sh/basics/
  enterShell = ''
    # Create a symlink to the Python virtual environment for IDE compatibility
    if [ ! -L "$DEVENV_ROOT/venv" ]; then
      ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/venv"
    fi
  '';

  # See https://github.com/cachix/devenv/issues/935
#   devcontainer.enable = true;
#   devcontainer.settings.remoteUser = "root";
#   devcontainer.settings.updateRemoteUserUID = false;

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
