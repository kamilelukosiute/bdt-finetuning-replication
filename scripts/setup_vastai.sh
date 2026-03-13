#!/bin/bash
# Vast.ai instance setup script
# Run this as root BEFORE launching Claude Code
#
# Usage (after SSH'ing into Vast.ai as root):
#   bash scripts/setup_vastai.sh
#   su - kamile
#   cd /workspace/bdt-finetuning-replication
#   claude  # launches Claude Code with full permissions
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

# 2. Give user ownership of workspace
chown -R "$USERNAME:$USERNAME" /workspace/
echo "Set /workspace/ ownership to $USERNAME"

# 3. Install Claude Code if not present
if ! command -v claude &>/dev/null; then
    npm install -g @anthropic-ai/claude-code
    echo "Installed Claude Code"
else
    echo "Claude Code already installed"
fi

# 4. Install deps into the container's environment
pip install evo2 huggingface_hub tqdm pandas matplotlib seaborn scipy 2>/dev/null || true

echo ""
echo "=== Done! ==="
echo "Now run:"
echo "  su - $USERNAME"
echo "  cd /workspace/bdt-finetuning-replication"
echo "  claude"
