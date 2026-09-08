-- lap library schema as created by lap 0.3.1 (user_version 16), the tables the export writes
CREATE TABLE acollections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
CREATE TABLE acollections_files (
            collection_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            added_at INTEGER NOT NULL,
            PRIMARY KEY (collection_id, file_id),
            FOREIGN KEY (collection_id) REFERENCES acollections(id) ON DELETE CASCADE,
            FOREIGN KEY (file_id) REFERENCES afiles(id) ON DELETE CASCADE
        );
CREATE TABLE afiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            name_pinyin TEXT,
            size INTEGER NOT NULL,
            file_type INTEGER,
            format_label TEXT,
            created_at INTEGER,
            modified_at INTEGER,
            inode INTEGER,
            taken_date INTEGER,
            width INTEGER,
            height INTEGER,
            duration INTEGER,
            is_favorite INTEGER,
            rating INTEGER NOT NULL DEFAULT 0,
            culling_flag INTEGER NOT NULL DEFAULT 0,
            rotate INTEGER,
            comments TEXT,
            has_tags INTEGER,
            has_faces INTEGER DEFAULT 0,
            e_make TEXT,
            e_model TEXT,
            e_date_time TEXT,
            e_software TEXT,
            e_artist TEXT,
            e_copyright TEXT,
            e_description TEXT,
            e_lens_make TEXT,
            e_lens_model TEXT,
            e_exposure_bias TEXT,
            e_exposure_time TEXT,
            e_f_number TEXT,
            e_focal_length TEXT,
            e_iso_speed TEXT,
            e_flash TEXT,
            e_orientation INTEGER,
            gps_latitude REAL,
            gps_longitude REAL,
            gps_altitude REAL,
            geo_name TEXT,
            geo_admin1 TEXT,
            geo_admin2 TEXT,
            geo_cc TEXT,
            embeds BLOB,
            last_scan_time INTEGER DEFAULT 0,
            content_identifier TEXT,
            media_subtype TEXT,
            live_photo_video_id INTEGER,
            motion_photo_offset INTEGER,
            FOREIGN KEY (folder_id) REFERENCES afolders(id) ON DELETE CASCADE
        );
CREATE TABLE afolders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            album_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at INTEGER,
            modified_at INTEGER,
            is_favorite INTEGER,
            is_excluded_from_search INTEGER DEFAULT 0,
            has_subfolders INTEGER,
            inode INTEGER,
            FOREIGN KEY (album_id) REFERENCES albums(id) ON DELETE CASCADE
        );
CREATE TABLE albums (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at INTEGER,
            modified_at INTEGER,
            display_order_id INTEGER,
            cover_file_id INTEGER,
            description TEXT,
            indexed INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0,
            skipped_count INTEGER NOT NULL DEFAULT 0,
            skipped_size INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0,
            failed_size INTEGER NOT NULL DEFAULT 0,
            merged_count INTEGER NOT NULL DEFAULT 0,
            merged_size INTEGER NOT NULL DEFAULT 0,
            last_scan_time INTEGER DEFAULT 0
        );
CREATE TABLE athumbs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL UNIQUE,
            error_code INTEGER NOT NULL,
            thumb_data BLOB,
            thumb_key TEXT,
            thumb_mtime INTEGER,
            thumb_size INTEGER,
            updated_at INTEGER,
            FOREIGN KEY (file_id) REFERENCES afiles(id) ON DELETE CASCADE
        );
CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            bbox TEXT,
            embedding BLOB,
            person_id INTEGER,
            created_at INTEGER,
            FOREIGN KEY (file_id) REFERENCES afiles(id) ON DELETE CASCADE,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
        );
CREATE TABLE persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            cover_face_id INTEGER,
            thumbnail BLOB,
            created_at INTEGER
        );
CREATE INDEX idx_acollections_files_collection_added ON acollections_files(collection_id, added_at DESC, file_id);
CREATE INDEX idx_acollections_files_file ON acollections_files(file_id);
CREATE INDEX idx_acollections_sort ON acollections(sort_order, id);
CREATE INDEX idx_afiles_content_identifier ON afiles(content_identifier);
CREATE INDEX idx_afiles_culling_flag ON afiles(culling_flag);
CREATE INDEX idx_afiles_file_type ON afiles(file_type);
CREATE INDEX idx_afiles_folder_id ON afiles(folder_id);
CREATE INDEX idx_afiles_folder_name_nocase
                     ON afiles(folder_id, name COLLATE NOCASE);
CREATE INDEX idx_afiles_geo_admin1 ON afiles(geo_admin1);
CREATE INDEX idx_afiles_geo_admin2 ON afiles(geo_admin2);
CREATE INDEX idx_afiles_geo_cc ON afiles(geo_cc);
CREATE INDEX idx_afiles_geo_name ON afiles(geo_name);
CREATE INDEX idx_afiles_gps_coordinates
         ON afiles(gps_latitude, gps_longitude)
         WHERE gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL;
CREATE INDEX idx_afiles_has_faces ON afiles(has_faces);
CREATE INDEX idx_afiles_has_tags ON afiles(has_tags);
CREATE INDEX idx_afiles_is_favorite ON afiles(is_favorite);
CREATE INDEX idx_afiles_last_scan_time ON afiles(last_scan_time);
CREATE INDEX idx_afiles_lens_make_model ON afiles(e_lens_make, e_lens_model);
CREATE INDEX idx_afiles_live_photo_video_id ON afiles(live_photo_video_id);
CREATE INDEX idx_afiles_make_model ON afiles(e_make, e_model);
CREATE INDEX idx_afiles_name ON afiles(name);
CREATE INDEX idx_afiles_name_pinyin ON afiles(name_pinyin);
CREATE INDEX idx_afiles_rating ON afiles(rating);
CREATE INDEX idx_afiles_taken_date ON afiles(taken_date);
CREATE INDEX idx_afolders_album_id ON afolders(album_id);
CREATE INDEX idx_afolders_album_inode ON afolders(album_id, inode);
CREATE INDEX idx_afolders_is_excluded_from_search ON afolders(is_excluded_from_search);
CREATE INDEX idx_afolders_is_favorite ON afolders(is_favorite);
CREATE INDEX idx_afolders_name ON afolders(name);
CREATE INDEX idx_afolders_path ON afolders(path);
CREATE INDEX idx_albums_name ON albums(name);
CREATE INDEX idx_albums_path ON albums(path);
CREATE INDEX idx_athumbs_file_id ON athumbs(file_id);
CREATE INDEX idx_athumbs_thumb_key ON athumbs(thumb_key);
CREATE INDEX idx_faces_file_id ON faces(file_id);
CREATE INDEX idx_faces_person_id ON faces(person_id);
CREATE INDEX idx_persons_name ON persons(name);
CREATE UNIQUE INDEX uidx_afiles_folder_id_name
            ON afiles(folder_id, name);
