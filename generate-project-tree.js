#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

/**
 * Simple gitignore pattern matcher
 * Handles basic patterns like *, **, directory/, file extensions, etc.
 */
class GitIgnoreParser {
  constructor(gitignoreContent) {
    this.patterns = this.parseGitignore(gitignoreContent);
  }

  parseGitignore(content) {
    return content
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"))
      .map((pattern) => {
        // Convert gitignore patterns to regex
        let regexPattern = pattern
          .replace(/\./g, "\\.")
          .replace(/\*/g, "[^/]*")
          .replace(/\*\*/g, ".*");

        // Handle directory patterns (ending with /)
        if (pattern.endsWith("/")) {
          regexPattern = "^" + regexPattern.slice(0, -1) + "(/.*)?$";
        } else {
          regexPattern = "^" + regexPattern + "$";
        }

        return {
          original: pattern,
          regex: new RegExp(regexPattern),
          isDirectory: pattern.endsWith("/"),
        };
      });
  }

  isIgnored(filePath, isDirectory = false) {
    // Normalize path separators
    const normalizedPath = filePath.replace(/\\/g, "/");

    for (const pattern of this.patterns) {
      // Check if pattern matches
      if (pattern.regex.test(normalizedPath)) {
        return true;
      }

      // Also check against just the filename/dirname
      const basename = path.basename(normalizedPath);
      if (pattern.regex.test(basename)) {
        return true;
      }

      // For directory patterns, check if any parent directory matches
      if (pattern.isDirectory) {
        const pathParts = normalizedPath.split("/");
        for (let i = 0; i < pathParts.length; i++) {
          const partialPath = pathParts.slice(0, i + 1).join("/");
          if (pattern.regex.test(partialPath + "/")) {
            return true;
          }
        }
      }
    }

    return false;
  }
}

/**
 * Generate project tree structure
 */
function generateTree(rootPath) {
  const gitignorePath = path.join(rootPath, ".gitignore");
  let gitignoreParser = null;

  // Read .gitignore if it exists
  if (fs.existsSync(gitignorePath)) {
    const gitignoreContent = fs.readFileSync(gitignorePath, "utf8");
    gitignoreParser = new GitIgnoreParser(gitignoreContent);
  }

  function buildTree(currentPath, relativePath = "") {
    const stats = fs.statSync(currentPath);
    const name = path.basename(currentPath);

    if (stats.isDirectory()) {
      const node = {
        name: name || path.basename(rootPath),
        type: "directory",
        path: relativePath || ".",
        children: [],
      };

      // Check if this directory should be ignored
      if (
        gitignoreParser &&
        relativePath &&
        gitignoreParser.isIgnored(relativePath, true)
      ) {
        node.ignored = true;
        return node; // Don't recurse into ignored directories
      }

      try {
        const entries = fs.readdirSync(currentPath);

        for (const entry of entries) {
          // Skip .git directory by default
          if (entry === ".git") continue;

          const entryPath = path.join(currentPath, entry);
          const entryRelativePath = relativePath
            ? path.join(relativePath, entry)
            : entry;

          try {
            const childNode = buildTree(entryPath, entryRelativePath);
            if (childNode) {
              node.children.push(childNode);
            }
          } catch (error) {
            // Skip files/directories that can't be accessed
            console.warn(
              `Warning: Could not access ${entryPath}: ${error.message}`
            );
          }
        }

        // Sort children: directories first, then files, alphabetically
        node.children.sort((a, b) => {
          if (a.type !== b.type) {
            return a.type === "directory" ? -1 : 1;
          }
          return a.name.localeCompare(b.name);
        });
      } catch (error) {
        console.warn(
          `Warning: Could not read directory ${currentPath}: ${error.message}`
        );
      }

      return node;
    } else {
      // It's a file
      const node = {
        name: name,
        type: "file",
        path: relativePath,
      };

      // Check if this file should be ignored
      if (gitignoreParser && gitignoreParser.isIgnored(relativePath, false)) {
        node.ignored = true;
        // Remove extension from ignored files as requested
        const ext = path.extname(name);
        if (ext) {
          node.name = path.basename(name, ext);
        }
      }

      return node;
    }
  }

  return buildTree(rootPath);
}

/**
 * Main execution
 */
function main() {
  const rootPath = process.cwd();
  console.log(`Generating project tree for: ${rootPath}`);

  try {
    const tree = generateTree(rootPath);

    // Add metadata
    const result = {
      generated: new Date().toISOString(),
      rootPath: rootPath,
      projectName: path.basename(rootPath),
      tree: tree,
    };

    const outputPath = path.join(rootPath, "project_tree.json");
    fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));

    console.log(`✅ Project tree generated successfully!`);
    console.log(`📁 Output saved to: ${outputPath}`);
    console.log(`📊 Total items processed: ${countNodes(tree)}`);
  } catch (error) {
    console.error("❌ Error generating project tree:", error.message);
    process.exit(1);
  }
}

/**
 * Helper function to count nodes in the tree
 */
function countNodes(node) {
  let count = 1;
  if (node.children) {
    for (const child of node.children) {
      count += countNodes(child);
    }
  }
  return count;
}

// Run the script
if (require.main === module) {
  main();
}
