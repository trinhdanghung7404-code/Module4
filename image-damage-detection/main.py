from product_pipeline import add_product
from return_pipeline import analyze_return_product


def menu():
    while True:
        print("\n==============================")
        print(" Damage Assessment System")
        print("==============================")
        print("1. Add New Product")
        print("2. Analyze Return Product")
        print("0. Exit")

        choice = input("Choose: ")

        if choice == "1":
            add_product()

        elif choice == "2":
            analyze_return_product()

        elif choice == "0":
            print("Goodbye!")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    menu()