{
  description = "Application packaged using poetry2nix";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        inherit (poetry2nix.legacyPackages.${system}) mkPoetryApplication;
        inherit (poetry2nix.legacyPackages.${system}) mkPoetryEnv;
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        packages = {
          myapp = mkPoetryApplication { projectDir = self; };
          default = self.packages.${system}.myapp;
        };

        devShells.default = pkgs.mkShell {
          name = "poetry-openmax";
          # LD_LIBRARY_PATH = "${nixpkgs.stdenv.cc.cc.lib}/lib";
          packages =
            # [ (mkPoetryEnv {projectDir = self;}) poetry2nix.packages.${system}.poetry pkgs.black pkgs.pyright];
            [
              poetry2nix.packages.${system}.poetry
              pkgs.black
              pkgs.pyright
              pkgs.stdenv.cc.cc.lib
              pkgs.python310Packages.numpy
              pkgs.python310Packages.scikit-learn-extra
			  pkgs.python310Packages.mypy
            ];
        };

        myapp.${system}.default = {
          type = "app";
          program = "${self.packages.${system}.myapp}";
        };
      });
}
