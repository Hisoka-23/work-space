package Recursion;

import java.util.Scanner;

//Q4. Implement a program to find factorial of the given number?

class Demo4 {
	static int factorial(int n) {
		if(n == 1) {
			return 1;
		} else {
			return n*factorial(n-1);
		}
	}
}

public class Factorial {
	
	public static void main(String[] args) {
		Scanner obj = new Scanner(System.in);
		
		int n = obj.nextInt();
		
		System.out.println(Demo4.factorial(n));
	}

}
