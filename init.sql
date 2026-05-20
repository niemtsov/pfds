USE my_database;

DROP TABLE IF EXISTS titanic;

CREATE TABLE titanic (
    PassengerId INT PRIMARY KEY,
    Survived INT,
    Pclass INT,
    Name VARCHAR(255),
    Sex VARCHAR(10),
    Age FLOAT NULL,
    SibSp INT,
    Parch INT,
    Ticket VARCHAR(100),
    Fare DECIMAL(10, 4),
    Cabin VARCHAR(50) NULL,
    Embarked CHAR(1) NULL
);

LOAD DATA INFILE '/var/lib/mysql-files/titanic.csv'
INTO TABLE titanic
FIELDS TERMINATED BY ',' 
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(PassengerId, Survived, Pclass, Name, Sex, @v_Age, SibSp, Parch, Ticket, Fare, @v_Cabin, @v_Embarked)
SET 
    Age = NULLIF(@v_Age, ''),
    Cabin = NULLIF(@v_Cabin, ''),
    Embarked = NULLIF(@v_Embarked, '');