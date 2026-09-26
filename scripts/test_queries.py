from src.tools.structured_query import (
    dataset_statistics,
    find_by_cas,
    find_by_chemical,
    find_by_company,
    find_discontinued_between,
)


print("=== DATASET COUNTS ===")
print(dataset_statistics())

print("\n=== CHEMICAL SEARCH ===")
print(find_by_chemical("Titanium dioxide").head(10).to_string(index=False))

print("\n=== COMPANY SEARCH ===")
print(find_by_company("New Avon LLC").head(10).to_string(index=False))

print("\n=== CAS SEARCH ===")
print(find_by_cas("75-07-0").head(20).to_string(index=False))

print("\n=== DISCONTINUED IN 2024 ===")
print(find_discontinued_between("2024-01-01", "2025-01-01").head(20).to_string(index=False))