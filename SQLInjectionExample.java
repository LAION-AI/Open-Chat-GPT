// Security Vulnerability #3757
// The below code is for the security vulnerability that has been programmed by me in java itself. You can simply convert my code to your preferred programming language and then edit it and after that use it.

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Scanner;

public class SQLInjectionExample {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter username: ");
        String username = scanner.nextLine();
        
        System.out.print("Enter password: ");
        String password = scanner.nextLine();

        try {
            // Connect to the database
            Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/mydatabase", "root", "password");
            Statement statement = connection.createStatement();

            // Vulnerable SQL query (user inputs are not sanitized)
            String query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'";

            ResultSet resultSet = statement.executeQuery(query);

            // Checking if login is successful
            if (resultSet.next()) {
                System.out.println("Login successful!");
            } else {
                System.out.println("Login failed!");
            }

            connection.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        scanner.close();
    }
}
