#!/bin/bash
# Vast.ai instance setup script
# Run this as root BEFORE launching Claude Code
#
# Usage as on-start script:
#   git clone https://github.com/kamilelukosiute/bdt-finetuning-replication.git /workspace/bdt-finetuning-replication && bash /workspace/bdt-finetuning-replication/scripts/setup_vastai.sh
#
# Then SSH in and run:
#   su - kamile
#   cd /workspace/bdt-finetuning-replication
#   claude --dangerously-skip-permissions
set -euo pipefail

USERNAME="kamile"

echo "=== Setting up Vast.ai instance ==="

# 1. Create user with sudo (idempotent)
if ! id "$USERNAME" &>/dev/null; then
    useradd -m -s /bin/bash "$USERNAME"
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    echo "Created user $USERNAME with passwordless sudo"
else
    echo "User $USERNAME already exists"
fi

# 2. Install Node.js if not present (needed for Claude Code)
if ! command -v node &>/dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi

# 3. Install Claude Code
if ! command -v claude &>/dev/null; then
    npm install -g @anthropic-ai/claude-code
    echo "Installed Claude Code"
else
    echo "Claude Code already installed"
fi

# 4. Install Python deps into the container's environment
pip install evo2 huggingface_hub tqdm pandas matplotlib seaborn scipy biopython 2>/dev/null || true

# 5. Add Node/npm bin to kamile's PATH (idempotent, in both .profile and .bashrc)
NODE_BIN=$(dirname "$(which node)")
for rc in /home/$USERNAME/.profile /home/$USERNAME/.bashrc; do
    if ! grep -q "$NODE_BIN" "$rc" 2>/dev/null; then
        echo "export PATH=\"$NODE_BIN:\$PATH\"" >> "$rc"
    fi
done
chown "$USERNAME:$USERNAME" /home/$USERNAME/.profile /home/$USERNAME/.bashrc

# 6. Set git config for the user (write directly to avoid sudo/HOME issues)
install -o "$USERNAME" -g "$USERNAME" -m 644 /dev/null /home/$USERNAME/.gitconfig
cat > /home/$USERNAME/.gitconfig <<'GITCFG'
[user]
	name = kamilelukosiute
	email = lukosiutekamile@gmail.com
[safe]
	directory = /workspace/bdt-finetuning-replication
GITCFG
chown "$USERNAME:$USERNAME" /home/$USERNAME/.gitconfig

# 7. Give user ownership of workspace (AFTER clone so repo is included)
chown -R "$USERNAME:$USERNAME" /workspace/
echo "Set /workspace/ ownership to $USERNAME"

echo ""
echo "=== Done! ==="
echo "Now SSH in and run:"
echo "  su - $USERNAME"
echo "  cd /workspace/bdt-finetuning-replication"
echo "  claude --dangerously-skip-permissions"
