-- 1. Create the Database (if it doesn't exist)
CREATE DATABASE IF NOT EXISTS face_db;

-- 2. Select the Database
USE face_db;

-- 3. Drop the old table (to ensure a clean slate)
DROP TABLE IF EXISTS authorized_faces;

-- 4. Create the new table with correct encryption columns
CREATE TABLE authorized_faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    embedding_cipher BLOB NOT NULL,       -- Stores the encrypted face embedding
    aes_nonce VARBINARY(64) NOT NULL,     -- Stores the initialization vector (IV)
    aes_tag VARBINARY(64) NOT NULL,       -- Stores the authentication tag (GCM)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Verify the table creation (Optional)
DESCRIBE authorized_faces;