#!/bin/bash
# Vast.ai instance setup script
# Run this as root BEFORE launching Claude Code
#
# Usage as on-start script:
#   git clone https://github.com/kamilelukosiute/bdt-finetuning-replication.git /workspace/bdt-finetuning-replication && bash /workspace/bdt-finetuning-replication/scripts/setup_vastai.sh
#
# Then SSH in and run:
#   runuser -u worker -- bash --login -c "cd /workspace/bdt-finetuning-replication && claude --dangerously-skip-permissions"
set -euo pipefail

USERNAME="worker"

echo "=== Setting up Vast.ai instance ==="

# 1. Create non-root user (idempotent)
# NOTE: Vast.ai containers have a cursed /etc/profile that sources /root/.bash_profile,
# so `su - user` always fails. Use `runuser` instead.
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

# 4. Install GitHub CLI
if ! command -v gh &>/dev/null; then
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt-get update && apt-get install -y gh
    echo "Installed GitHub CLI"
else
    echo "GitHub CLI already installed"
fi

# 5. Install Python deps into the container's environment
pip install evo2 huggingface_hub tqdm pandas matplotlib seaborn scipy biopython 2>/dev/null || true

# 5. Add Node/npm bin to user's PATH (idempotent)
NODE_BIN=$(dirname "$(which node)")
for rc in /home/$USERNAME/.profile /home/$USERNAME/.bashrc; do
    if ! grep -q "$NODE_BIN" "$rc" 2>/dev/null; then
        echo "export PATH=\"$NODE_BIN:\$PATH\"" >> "$rc"
    fi
done

# 6. Set git config for the user (write directly to avoid sudo/HOME issues)
cat > /home/$USERNAME/.gitconfig <<'GITCFG'
[user]
	name = kamilelukosiute
	email = lukosiutekamile@gmail.com
[safe]
	directory = /workspace/bdt-finetuning-replication
GITCFG

# 7. Give user ownership of everything they need
chown -R "$USERNAME:$USERNAME" /home/$USERNAME /workspace/
echo "Set ownership to $USERNAME"

echo ""
echo "=== Done! ==="
echo "SSH in and run:"
echo "  runuser -u $USERNAME -- bash --login -c 'cd /workspace/bdt-finetuning-replication && claude --dangerously-skip-permissions'"
