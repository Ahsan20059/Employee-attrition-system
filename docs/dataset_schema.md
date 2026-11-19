# Dataset schema

| Field | Inferred type | Role or definition |
|---|---|---|
| `Age` | int64 | Age in years as recorded in the source. |
| `DailyRate` | int64 | Recorded daily rate. Units are unspecified. |
| `DistanceFromHome` | int64 | Recorded commuting distance. Units are unspecified. |
| `HourlyRate` | int64 | Recorded hourly rate. Units are unspecified. |
| `MonthlyIncome` | int64 | Recorded monthly income. Currency is unspecified. |
| `MonthlyRate` | int64 | Recorded monthly rate field. Definition and units are unspecified. |
| `NumCompaniesWorked` | int64 | Number of employers reported in the source field. |
| `PercentSalaryHike` | int64 | Recorded salary increase percentage. |
| `TotalWorkingYears` | int64 | Recorded years of working experience. |
| `TrainingTimesLastYear` | int64 | Recorded training instances in the prior year. |
| `YearsAtCompany` | int64 | Recorded company tenure in years. |
| `YearsInCurrentRole` | int64 | Recorded years in the current role. |
| `YearsSinceLastPromotion` | int64 | Recorded years since the last promotion. |
| `YearsWithCurrManager` | int64 | Recorded years with the current manager. |
| `Education` | int64 | Ordinal education level, 1–5. |
| `JobLevel` | int64 | Ordinal level of the recorded position, 1–5. |
| `StockOptionLevel` | int64 | Ordinal stock option level, 0–3. |
| `PerformanceRating` | int64 | Ordinal performance rating. |
| `EnvironmentSatisfaction` | int64 | Ordinal environment satisfaction rating, 1–4. |
| `JobInvolvement` | int64 | Ordinal job involvement rating, 1–4. |
| `JobSatisfaction` | int64 | Ordinal job satisfaction rating, 1–4. |
| `RelationshipSatisfaction` | int64 | Ordinal relationship satisfaction rating, 1–4. |
| `WorkLifeBalance` | int64 | Ordinal work–life balance rating, 1–4. |
| `BusinessTravel` | object | Categorical frequency of business travel. |
| `Department` | object | Organizational department. |
| `EducationField` | object | Field of education. |
| `Gender` | object | Source-provided audit field excluded from dashboards and model inputs. |
| `JobRole` | object | Recorded job role. |
| `MaritalStatus` | object | Source-provided audit field excluded from dashboards and model inputs. |
| `OverTime` | object | Source overtime flag, Yes or No. |
| `Attrition` | int8 | Historical target. 1 records a departure and 0 records a stayed label. |
| `Record` | Application-derived | Stable derived key for this normalized source record. |
| `Probability` | Application-derived | Selected model’s estimated attrition probability for the record. |
| `RiskScore` | Application-derived | Estimated probability multiplied by 100. |
| `RiskBand` | Application-derived | Display category derived from the selected risk boundaries. |
