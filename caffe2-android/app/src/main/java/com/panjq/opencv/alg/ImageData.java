package com.panjq.opencv.alg;

import android.graphics.Bitmap;

/**
 * Created by panjq1 on 2017/10/23.
 */

public class ImageData {
   // public Bitmap bitmap;
    public int[] pixels;
    public int w;
    public int h;

    ImageData(){
    }

    ImageData(Bitmap bitmap){
        this.w = bitmap.getWidth();
        this.h = bitmap.getHeight();
        //将bitmap类型转为int数组
        this.pixels = new int[this.w * this.h];
        bitmap.getPixels(this.pixels, 0, this.w, 0, 0, this.w, this.h);
    }

    public  Bitmap getBitmap( ){
        //int数组转为bitmap类型。
        Bitmap desImage=Bitmap.createBitmap(this.w,this.h,Bitmap.Config.ARGB_8888);
        desImage.setPixels(this.pixels,0,this.w,0,0,this.w,this.h);
        return desImage;
    }

    public  ImageData  getImageData(Bitmap bitmap){
        this.w = bitmap.getWidth();
        this.h = bitmap.getHeight();
        this.pixels = new int[w * h];
        bitmap.getPixels( this.pixels, 0, w, 0, 0, w, h);
        return this;
    }
}
