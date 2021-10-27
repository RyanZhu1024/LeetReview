package leet;

/**
 * Created by RyanZhu on 2/7/16.
 */
public class Lyft {

    public static void main(String[] args) {
        double aLat = 12,aLon = 21,bLat = 31,bLon = 13,cLat = 44,cLon = 55,dLat = 0,dLon = 0;
        GeoLocation a=new GeoLocation(aLat,aLon);
        GeoLocation b=new GeoLocation(bLat,bLon);
        GeoLocation c=new GeoLocation(cLat,cLon);
        GeoLocation d=new GeoLocation(dLat,dLon);
        // only two paths can be as candidates. ACDB and CABD. Because eath path will go through AC and BD,
        // the only distance that matters is AB and CD
        float ab=distance(a,b);
        float cd=distance(c,d);
        if(ab>cd){
            System.out.println("The shortest path is ACDB");
        }else{
            System.out.println("The shortest path is CABD");
        }
    }

    public static float distance(GeoLocation start, GeoLocation end) {
        double earthRadius = 3958.75;
        double yLatitude = end.getLatitude();
        double xLatitude = start.getLatitude();
        double dLatitude = Math.toRadians(yLatitude - xLatitude);
        double yLongitude = end.getLongitude();
        double xLongitude = start.getLongitude();
        double dLongitude = Math.toRadians(yLongitude - xLongitude);
        double a = Math.sin(dLatitude / 2) * Math.sin(dLatitude / 2)
                + Math.cos(Math.toRadians(xLatitude))
                * Math.cos(Math.toRadians(yLatitude)) * Math.sin(dLongitude / 2)
                * Math.sin(dLongitude / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        double dist = earthRadius * c;

        int meterConversion = 1609;

        return new Float(dist * meterConversion).floatValue();
    }
}
