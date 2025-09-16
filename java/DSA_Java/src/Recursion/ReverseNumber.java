package Recursion;

import java.util.Scanner;

class Demo6{
	static int reverse(int n, int len) {
		if(n == 0)
			return 0;
		else
			return ((n%10)*(int)Math.pow(10, len-1)) + reverse(n/10, --len);
	}
}

public class ReverseNumber {

	public static void main(String[] args) {
		Scanner obj = new Scanner(System.in);
		String n = obj.nextLine();
		System.out.println(Demo6.reverse(Integer.parseInt(n), n.length()));//reverse of 'n'
	}
	
}
