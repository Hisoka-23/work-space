package Arrays;

import java.util.Arrays;

class Demo222{
	static int[] deleteElementAlLocation(int a[], int location) {
		int k=0,i,b[] = new int[a.length-1];
		for(i=0; i<a.length; i++) {
			if(i==location) {
				System.out.println(a[i]);
				continue;
			}
			b[k++] = a[i];
		}
		return b;
	}
	
	static int[] deleteAll(int a[]) {
		int b[] = new int [0];
		return b;
		
		//a=new int[0];
		//return a;
	}
	
	static int[] deleteElement(int a[], int element) {
		int index=-1,i,k=0;
		for(i=0;i<a.length;i++) {
			if(a[i] == element) {
				index = i;
				break;
			}
		}
		if(index!=-1) {
			int b[] = new int[a.length-1];
			for(i=0;i<a.length; i++) {
				if(i==index) {
					continue;
				}
				b[k++]=a[i];
			}
			return b;
		}
		return a;
	}
}

public class Delete {

	public static void main(String[] args) {
		int a[] = {10, 11, 12, 13, 15, 14, 16};
		System.out.println(Arrays.toString(a));
//		System.out.println(Arrays.toString(Demo222.deleteElementAlLocation(a,5)));
//		System.out.println(Arrays.toString(Demo222.deleteAll(a)));
		System.out.println(Arrays.toString(Demo222.deleteElement(a, 13)));
	}
	
}
