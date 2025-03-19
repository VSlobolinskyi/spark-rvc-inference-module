// generate-structure.js
const fs = require("fs");
const path = require("path");

const targetDir = path.join(__dirname, "./");
const outputFile = path.join(__dirname, "struct.txt");

function generateStructure(dir, indent = "") {
  let structure = "";
  // Read directory items
  const items = fs.readdirSync(dir);
  items.forEach((item, index) => {
    // Skip .git directories
    if (item === ".git") return;

    const itemPath = path.join(dir, item);
    const stats = fs.statSync(itemPath);
    // Use pointer symbols to show tree structure
    const pointer = index === items.length - 1 ? "└── " : "├── ";
    structure += indent + pointer + item + "\n";
    if (stats.isDirectory()) {
      // Increase indent: if last, add space, otherwise add vertical line
      const extension = index === items.length - 1 ? "    " : "│   ";
      structure += generateStructure(itemPath, indent + extension);
    }
  });
  return structure;
}

try {
  const structureText = generateStructure(targetDir);
  fs.writeFileSync(outputFile, structureText, "utf-8");
  console.log(`Folder structure saved to ${outputFile}`);
} catch (error) {
  console.error("Error generating folder structure:", error);
}
