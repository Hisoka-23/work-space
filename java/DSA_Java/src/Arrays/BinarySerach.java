package Arrays;

import java.util.Arrays;
import java.util.Scanner;

class Demo11{
	int binarySerach(int a[],int l, int h, int key) {
		
		
		while(l<=h) {
			int mid = (l+h)/2;
			if(a[mid] == key) {
				return mid;
			} else if(a[mid] < key) {
				return binarySerach(a, mid+1, h, key);
			} else {
				return binarySerach(a, l, mid-1, key);
			}
			
			
		}
		return -1;
		
	}
}

public class BinarySerach {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter the size of array");
		int size = obj.nextInt();
		
		System.out.println("Enter "+size+" elements in array");
		int arr[] = new int[size];
		
		for(int i=0; i<arr.length; i++) {
			arr[i] = obj.nextInt();
		}
		
		System.out.println("Enter the key value : ");
		int key = obj.nextInt();
		
		Arrays.sort(arr);
		
		Demo11 d = new Demo11();
		
		System.out.println(d.binarySerach(arr,0,(arr.length-1)/2, key));
		
	}
	
}
