package com.panjq.opencv.alg;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;

import com.panjq.opencv.opencvdemo.LogUtil;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import org.opencv.core.Mat;
import org.opencv.android.Utils;

/**
 * Created by panjq1 on 2017/10/22.
 */

public class ImagePro {
    private Context context;
    private static final String    TAG = "ImagePro:";
    static {
       // System.loadLibrary("imagePro-lib");
        //System.loadLibrary("opencv_java3");
    }
    public static  native void caffe2init(AssetManager mgr);
    public native void caffe2inference(long matAddrSrcImage, long matAddrDestImage);
    public ImagePro(Context context){
        this.context = context;
    }

    /**
     *
     */
    public Bitmap predictor(Bitmap origImage) {
        Log.i(TAG, "called JNI:jniImagePro3 ");
        int w=origImage.getWidth();
        int h=origImage.getHeight();
        Mat origMat = new Mat();
        Mat destMat = new Mat();
        Utils.bitmapToMat(origImage, origMat);
        caffe2inference(origMat.getNativeObjAddr(), destMat.getNativeObjAddr());
        int dest_W=destMat.width();
        int dest_H=destMat.height();
        Bitmap bitImage = Bitmap.createBitmap(dest_W, dest_H, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(destMat, bitImage);
        Log.i(TAG, "jniImagePro3 called successfully");
        return bitImage;
    }

    //图片保存
    public void saveBitmap(Bitmap b){
//        String path = Environment.getExternalStorageDirectory().getAbsolutePath()+"/3DLUT/";
        String path =Environment.getExternalStorageDirectory().getPath()+"/DCIM/3DLUT/";

        File folder=new File(path);
        if(!folder.exists()&&!folder.mkdirs()){
            LogUtil.e("无法保存图片");
            return;
        }
        long dataTake = System.currentTimeMillis();
        final String jpegName=path+ dataTake +".jpg";
        LogUtil.e("jpegName = "+jpegName);
        try {
            FileOutputStream fout = new FileOutputStream(jpegName);
            BufferedOutputStream bos = new BufferedOutputStream(fout);
            b.compress(Bitmap.CompressFormat.JPEG, 100, bos);
            bos.flush();
            bos.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        // 最后通知图库更新
        context.sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, Uri.parse("file://" + jpegName)));
    }

}
