package Recursion;

import java.util.Scanner;

class Demo11{
	static int reverse(int n, int len) {
		if(n ==  0) {
			return 0;
		} else {
			return ((n%10)*(int)Math.pow(10, len-1)) + reverse(n/10, --len);
		}
	}
}

public class reverseOfNumber {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter the Number...");
		
		String str = obj.nextLine();
		
		System.out.println(Demo11.reverse(Integer.parseInt(str), str.length()));
		
	}
	
}
