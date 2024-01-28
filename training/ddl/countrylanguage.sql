CREATE TABLE countrylanguage (
    countrycode character(3) NOT NULL,
    language text NOT NULL,
    isofficial boolean NOT NULL,
    percentage real NOT NULL,
    PRIMARY KEY(countrycode, language),
    CONSTRAINT fk_language_country FOREIGN KEY(countrycode) REFERENCES country(code)
);
