package edu.gatech.dbclass.t4r;

import java.math.BigDecimal;

public class Accessory implements PostProcessingEnabler.PostProcessable {

    private String description;
    private int quantity;
    private BigDecimal voltRating;
    private BigDecimal ampRating;
    private String batteryType;

    public Accessory(String description, int quantity) {
        this.description = description;
        this.quantity = quantity;
        this.description = createLongDescription();
    }
    public Accessory(BigDecimal voltRating, BigDecimal ampRating, String description, int quantity, String batteryType) {
        this.description = description;
        this.quantity = quantity;
        this.voltRating = voltRating;
        this.ampRating = ampRating;
        this.batteryType = batteryType;
        this.description = createLongDescription();
    }

    public String getDescription(){return description;}
    public int getQuantity(){return quantity;}
    public BigDecimal getAmpRating() { return ampRating; }
    public BigDecimal getVoltRating() { return voltRating; }
    public String getBatteryType(){ return batteryType; }

    /*
        If [acc-desc] IS “ Batteries ”
        Accessory-Description = [acc-quantity] + [volt-rating] + “ Volt ” + [amp-rating] + ” Amp ” + [battery-type] +  [acc-desc]
        If [acc-desc] IS NOT “ Batteries ”
        Accessory-Description = [acc-quantity] +  [acc-desc]
    */

    private String createLongDescription() {
        if (description.equalsIgnoreCase("Batteries")) {
            return quantity + " " + voltRating + " Volt " + ampRating + " Amp " + batteryType + " " + description;
        }
        else {
            return quantity + " " + description;
        }
    }

    private String createDescriptionForInsert() {
        if (description.equalsIgnoreCase("Batteries")) {
            return voltRating + " Volt " + ampRating + " Amp " + batteryType + " " + description;
        }
        else {
            return description;
        }
    }

    @Override
    public void postProcess() {
        this.description = createDescriptionForInsert();
    }
}
