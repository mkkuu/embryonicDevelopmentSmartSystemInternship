-- Active: 1759751643894@@127.0.0.1@5432@embryo
-- 0. Drop existing elements (safe cleanup)
DROP TRIGGER IF EXISTS trg_create_user_auth ON users;

DROP FUNCTION IF EXISTS create_user_auth ();

DROP TABLE IF EXISTS embryo CASCADE;

DROP TABLE IF EXISTS user_auth CASCADE;

DROP TABLE IF EXISTS users CASCADE;

DROP TYPE IF EXISTS user_role;

DROP TYPE IF EXISTS gender_type;

DROP TYPE IF EXISTS pgt_a_type;

-- 1. Create enum types
CREATE TYPE user_role AS ENUM ('Admin', 'Doctor');

CREATE TYPE gender_type AS ENUM ('Male', 'Female');

CREATE TYPE pgt_a_type AS ENUM ('Euploid', 'Aneuploid');

-- 2. Create users table with optimized constraints
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL ,
    last_name VARCHAR(50) NOT NULL ,
    birthday DATE CHECK (birthday <= CURRENT_DATE),
    contact VARCHAR(50) UNIQUE NOT NULL ,
    address TEXT,
    gender gender_type,
    role user_role NOT NULL,
    global_access BOOLEAN DEFAULT FALSE
);

-- 3. Create user_auth table with optimized structure
CREATE TABLE user_auth (
    user_id INT NOT NULL UNIQUE,
    username VARCHAR(50) NOT NULL ,
    password TEXT NOT NULL ,
    CONSTRAINT user_auth_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

-- 4. Function to generate random password
CREATE OR REPLACE FUNCTION generate_random_password(length INT DEFAULT 12)
RETURNS TEXT AS $$
DECLARE
    chars TEXT := 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()';
    result TEXT := '';
BEGIN
    FOR i IN 1..length LOOP
        result := result || substr(chars, floor(random() * length(chars) + 1)::int, 1);
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- 5. Trigger function to auto-insert into user_auth
CREATE OR REPLACE FUNCTION create_user_auth()
RETURNS TRIGGER AS $$
DECLARE
    random_password TEXT;
BEGIN
    random_password := generate_random_password(12);

    INSERT INTO user_auth(user_id, username, password)
    VALUES (NEW.user_id, NEW.contact, random_password);

    RAISE NOTICE 'Created user_auth for user_id %, username %, password %', NEW.user_id, NEW.contact, random_password;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 6. Trigger on users insert
CREATE TRIGGER trg_create_user_auth
AFTER INSERT ON users
FOR EACH ROW
EXECUTE FUNCTION create_user_auth();

-- 7. Create embryo table with optimized constraints
CREATE TABLE embryo (
    embryo_id SERIAL PRIMARY KEY,
    date DATE ,
    contact VARCHAR(50) ,
    blastocyst_grade VARCHAR(10),
    pgt_a_grade pgt_a_type,
    live_birth BOOLEAN,
    path TEXT,
    doctor_id INT NOT NULL,
    CONSTRAINT embryo_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES users (user_id) ON DELETE CASCADE
);

-- 8. Create optimized indexes for better performance
CREATE INDEX idx_users_role ON users (role);

CREATE INDEX idx_users_contact ON users (contact);

CREATE INDEX idx_user_auth_username ON user_auth (username);

CREATE INDEX idx_embryo_doctor_id ON embryo (doctor_id);

CREATE INDEX idx_embryo_date ON embryo (date);

CREATE INDEX idx_embryo_blastocyst_grade ON embryo (blastocyst_grade);

CREATE INDEX idx_embryo_pgt_a_grade ON embryo (pgt_a_grade);

CREATE INDEX idx_embryo_live_birth ON embryo (live_birth);

CREATE INDEX idx_embryo_doctor_date ON embryo (doctor_id, date);