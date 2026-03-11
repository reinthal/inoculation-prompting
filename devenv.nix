{pkgs, ...}: {
  packages = with pkgs; [azure-cli git opencode];
  
  env.SSL_CERT_FILE = "/etc/ssl/certs/ca-bundle.crt";
  languages.python = {
    uv.enable = true;
  };

  enterShell = ''
    echo "which uv: $(which uv)"
    echo " version: $(uv --version)"
  '';
}
