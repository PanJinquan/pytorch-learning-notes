package com.panjq.opencv.opencvdemo;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
//import android.hardware.camera2.CameraAccessException;

import com.panjq.opencv.alg.ImagePro;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    //调用系统相册-选择图片
    private static final int IMAGE = 1;
    private static final String    TAG = "MainActivity";
    private static  boolean bSHOW_SRCIMAGE  = true;

    static{
        Log.i(TAG, "opencv_java3 loading...");
        System.loadLibrary("opencv_java3");
        Log.i(TAG, "opencv_java3 loaded successfully");
        Log.i(TAG, "imagePro-lib loading...");
        System.loadLibrary("imagePro-lib");
        Log.i(TAG, "imagePro-lib loaded successfully");
    }

    private ImageView imageview;
    private Bitmap src_bitmap, dest_bitmap;
    private Button open_albumsBt, process_imageBt,contrast_imageBt;
    private AssetManager mgr;
    private String predictedClass = "none";


    static {
        File appDir = new File(Environment.getExternalStorageDirectory(), "OpencvDemo");
        if (!appDir.exists()) {
            appDir.mkdir();
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        verifyStoragePermissions(MainActivity.this);
        TextView tv = (TextView) findViewById(R.id.tv1);
        tv.setText("opencv-demo");

        mgr = getResources().getAssets();
        new SetUpNeuralNetwork().execute();

        imageview = (ImageView) findViewById(R.id.image_view);
        //src_bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.girl);
        try {
            src_bitmap = BitmapFactory.decodeStream(getAssets().open("houses.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        imageview.setImageBitmap(src_bitmap);

        //打开相册
        open_albumsBt = (Button) findViewById(R.id.open_albums);
        open_albumsBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //调用相册
                Intent intent = new Intent(Intent.ACTION_PICK,
                        android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, IMAGE);

            }
        });

        //图像处理
        process_imageBt = (Button) findViewById(R.id.process_image);
        process_imageBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Log.i(TAG, "onClick...");
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        ImagePro img=new ImagePro(MainActivity.this);
                        long T0 = System.currentTimeMillis();
                        Bitmap bitImage3 =img.predictor(src_bitmap);//通过OpenCV的getNativeObjAddr()把地址传递给底层JNI
                        long T1 = System.currentTimeMillis();
                        Log.e(TAG, "Run time,predictor="+(T1-T0)+"ms");
                        dest_bitmap =bitImage3;
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                showImage(dest_bitmap);
                            }
                        });

                    }
                }).start();
            }
        });

        //对比
        contrast_imageBt= (Button) findViewById(R.id.contrast_image);
        contrast_imageBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (bSHOW_SRCIMAGE){
                    showImage(src_bitmap);
                    bSHOW_SRCIMAGE=false;
                    contrast_imageBt.setText("原图");
                }else {
                    showImage(dest_bitmap);
                    bSHOW_SRCIMAGE=true;
                    contrast_imageBt.setText("效果图");
                }
            }
        });
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        //打开相册，获取图片路径
        if (requestCode == IMAGE && resultCode == Activity.RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            String[] filePathColumns = {MediaStore.Images.Media.DATA};
            Cursor c = getContentResolver().query(selectedImage, filePathColumns, null, null, null);
            c.moveToFirst();
            int columnIndex = c.getColumnIndex(filePathColumns[0]);
            String imagePath = c.getString(columnIndex);
            src_bitmap = BitmapFactory.decodeFile(imagePath);
            showImage(src_bitmap);
            c.close();
        }
    }
    //显示图片
    private void showImage(Bitmap bitmap){
        imageview.setImageBitmap(bitmap);
    }

    /**
     * 添加文件读写权限
     */
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE};

    public static void verifyStoragePermissions(Activity activity) {
        // Check if we have write permission
        int permission = ActivityCompat.checkSelfPermission(activity,
                Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE,
                    REQUEST_EXTERNAL_STORAGE);
        }
    }

    private class SetUpNeuralNetwork extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void[] v) {
            try {
                ImagePro.caffe2init(mgr);
                predictedClass = "Neural net loaded! Inferring...";
            } catch (Exception e) {
                Log.d(TAG, "Couldn't load neural network.");
            }
            return null;
        }
    }
}
