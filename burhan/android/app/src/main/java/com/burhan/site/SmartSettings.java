package com.burhan.site;

import android.content.Context;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;
import java.nio.charset.StandardCharsets;
import java.security.KeyStore;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;
import org.json.JSONObject;

/** Pairing credentials never enter exported application state or backup archives. */
final class SmartSettings {
    static final String DEFAULT_URL="https://merhaba-api.kiyan.finance";
    private static final String ALIAS="burhan-smart-pairing-v1";
    private static SecretKey key() throws Exception {
        KeyStore store=KeyStore.getInstance("AndroidKeyStore");store.load(null);
        if(store.containsAlias(ALIAS))return (SecretKey)store.getKey(ALIAS,null);
        KeyGenerator generator=KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES,"AndroidKeyStore");
        generator.init(new KeyGenParameterSpec.Builder(ALIAS,KeyProperties.PURPOSE_ENCRYPT|KeyProperties.PURPOSE_DECRYPT).setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).build());return generator.generateKey();
    }
    static String validUrl(String value) {
        android.net.Uri uri=android.net.Uri.parse(value);
        if(!"https".equals(uri.getScheme())||uri.getHost()==null||uri.getUserInfo()!=null||uri.getQuery()!=null||uri.getFragment()!=null||!(uri.getPath()==null||uri.getPath().isEmpty()||uri.getPath().equals("/")))throw new IllegalArgumentException("Backend requires an HTTPS origin");
        return value.replaceAll("/+$","");
    }
    static JSONObject get(Context context) throws Exception {
        android.content.SharedPreferences prefs=context.getSharedPreferences("smart-pairing",Context.MODE_PRIVATE);String encrypted=prefs.getString("token","");String token="";
        if(!encrypted.isEmpty()){
            String[] parts=encrypted.split(":",2);Cipher cipher=Cipher.getInstance("AES/GCM/NoPadding");cipher.init(Cipher.DECRYPT_MODE,key(),new GCMParameterSpec(128,Base64.decode(parts[0],Base64.NO_WRAP)));token=new String(cipher.doFinal(Base64.decode(parts[1],Base64.NO_WRAP)),StandardCharsets.UTF_8);
        }
        return new JSONObject().put("url",prefs.getString("url",DEFAULT_URL)).put("token",token);
    }
    static void set(Context context,String url,String token) throws Exception {
        url=validUrl(url);if(token.length()>4096)throw new IllegalArgumentException("Pairing credential too long");String encrypted="";
        if(!token.isEmpty()){Cipher cipher=Cipher.getInstance("AES/GCM/NoPadding");cipher.init(Cipher.ENCRYPT_MODE,key());encrypted=Base64.encodeToString(cipher.getIV(),Base64.NO_WRAP)+":"+Base64.encodeToString(cipher.doFinal(token.getBytes(StandardCharsets.UTF_8)),Base64.NO_WRAP);}
        if(!context.getSharedPreferences("smart-pairing",Context.MODE_PRIVATE).edit().putString("url",url).putString("token",encrypted).commit())throw new IllegalStateException("Cannot save pairing settings");
    }
}
