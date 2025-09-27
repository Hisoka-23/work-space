package Arrays;

import java.util.Scanner;

public class SortString {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter a String : ");
		
		String str = obj.nextLine();
		
		char[] s = str.toCharArray();
		
		char temp;
		for(int i=0; i<s.length; i++) {
			for(int j=i+1; j<s.length; j++) {
				if(s[i] > s[j]) {
					temp = s[i];
					s[i] = s[j];
					s[j] = temp;
				}
			}
		}
		
		System.out.print("sort string : ");
		for(int i=0; i<s.length; i++) {
			System.out.print(s[i]);
		}
		
	}
	
}
