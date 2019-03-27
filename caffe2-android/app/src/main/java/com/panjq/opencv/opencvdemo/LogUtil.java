package com.panjq.opencv.opencvdemo;

import android.util.Log;

/**
 * Created by lammy on 2017/11/15.
 */

public class LogUtil {
    public static boolean debug = true;
    final static String TAG = "lammy :  ";


    public  static void d(String tag , String meg)
    {
        if(debug)
        Log.d(TAG + tag ,meg);
    }
    public  static void e(String tag , String meg)
    {
        if(debug)
            Log.e(TAG + tag ,meg);
    }
    public  static void i(String tag , String meg)
    {
        if(debug)
            Log.i(TAG + tag ,meg);
    }
    public  static void v(String tag , String meg)
    {
        if(debug)
            Log.v(TAG + tag ,meg);
    }
    public  static void w(String tag , String meg)
    {
        if(debug)
            Log.w(TAG + tag ,meg);
    }


    public  static void d( String meg)
    {
        if(debug)
            Log.d(TAG  ,meg);
    }
    public  static void e( String meg)
    {
        if(debug)
            Log.e(TAG  ,meg);
    }
    public  static void i( String meg)
    {
        if(debug)
            Log.i(TAG  ,meg);
    }
    public  static void v(String meg)
    {
        if(debug)
            Log.v(TAG ,meg);
    }
    public  static void w( String meg)
    {
        if(debug)
            Log.w(TAG  ,meg);
    }

}
