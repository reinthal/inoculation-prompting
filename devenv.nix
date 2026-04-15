{pkgs, ...}: {
  packages = with pkgs; [age azure-cli git opencode];

  env.SSL_CERT_FILE = "/etc/ssl/certs/ca-bundle.crt";
  languages = {
    texlive = {
      base = pkgs.texliveFull;
      enable = true;
    };
    python = {
      uv.enable = true;
    };
  };

  enterShell = ''
    echo "which uv: $(which uv)"
    echo " version: $(uv --version)"
  '';
}
