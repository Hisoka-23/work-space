package Recursion;

import java.util.Scanner;

class Demo5{
	static int product(int a, int b) {
		if(a<b) {
			return product(b,a);
		} else if(b!=0) {
			return a+product(a,b-1);
		} else {
			return 0;
		}
	}
}


public class ProductOfTwoNo {
	
	public static void main(String[] args) {
		
		Scanner obj1 = new Scanner(System.in);
		
		Scanner obj2 = new Scanner(System.in);
		
		System.out.println("Enter a value");
		
		int a = obj1.nextInt();
		
		System.out.println("Enter b value");
		
		int b = obj2.nextInt();
		
		System.out.println(Demo5.product(a, b));
		
	}
	
}
