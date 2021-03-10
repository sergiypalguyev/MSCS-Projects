package edu.gatech.dbclass.t4r;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.util.List;

public class Tool implements Comparable<Tool> {
    private int toolID;
    private String shortDesc;
    private String longDesc;
    private String powerSource;
    private String subOption;
    private String subType;
    private BigDecimal rentalPrice;
    private BigDecimal depositPrice;
    private String type;
    private String material;
    private BigDecimal length;
    private BigDecimal width;
    private BigDecimal weight;
    private String manufacturer;
    private BigDecimal purchasePrice;

    public Tool(int toolID, String powerSource, String subOption, String subType,
                BigDecimal rentalPrice, BigDecimal depositPrice) {
        this.toolID = toolID;
        this.powerSource = powerSource;
        this.subOption = subOption;
        this.subType = subType;
        this.rentalPrice = rentalPrice;
        this.depositPrice = depositPrice;
        this.shortDesc = createShortDescription();
    }

    private String status;
    public String getStatus(){return status;}
    private Timestamp date;
    public Timestamp getDate(){return date;}
    private BigDecimal totalCost;
    public BigDecimal getTotalCost(){return totalCost;}
    public void setTotalCost(BigDecimal totalCost){this.totalCost = totalCost;}
    private BigDecimal rentalProfit;
    public BigDecimal getRentalProfit(){return rentalProfit;}
    public void setRentalProfit(BigDecimal rentalProfit){this.rentalProfit = rentalProfit;}

    public Tool(int toolID, String powerSource, String subOption, String subType,
                BigDecimal rentalPrice, BigDecimal depositPrice, String status, Timestamp date, BigDecimal totalCost, BigDecimal rentalProfit) {
        this.toolID = toolID;
        this.powerSource = powerSource;
        this.subOption = subOption;
        this.subType = subType;
        this.rentalPrice = rentalPrice;
        this.depositPrice = depositPrice;
        this.shortDesc = createShortDescription();
        this.status = status;
        this.date = date;
        this.totalCost = totalCost;
        this.rentalProfit = rentalProfit;
    }

    public Tool(int toolID, String type, String powerSource, String subOption, String subType,
                String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice) {
        this(toolID, powerSource, subOption, subType, rentalPrice, depositPrice);
        this.type = type;
        this.material = material;
        this.length = length;
        this.width = width;
        this.weight = weight;
        this.manufacturer = manufacturer;
        this.purchasePrice = purchasePrice;
    }

    public int getToolID() { return toolID; }

    public String getShortDesc() { return shortDesc; }

    public BigDecimal getRentalPrice() { return rentalPrice; }

    public BigDecimal getDepositPrice() { return depositPrice; }

    public String getType() { return type; }

    public String getSubOption(){return subOption;}

    public String getSubType(){return subType;}

    public String getPowerSource() { return powerSource; }

    public String getMaterial() { return material; }

    public BigDecimal getLength() { return length; }

    public BigDecimal getWidth() { return width; }

    public BigDecimal getWeight() { return weight; }

    public String getManufacturer() { return manufacturer; }

    public BigDecimal getPurchasePrice() { return purchasePrice; }

    public String getLongDesc(){return longDesc;}

    public void setLongDesc(String longDescription){this.longDesc = longDescription;}

    private String createShortDescription() {
        if (this.powerSource.equalsIgnoreCase("MANUAL")) {
            return this.subOption + " " + this.subType;
        } else {
            return this.powerSource + " " + this.subOption + " " + this.subType;
        }
    }

    @Override
    public int compareTo(Tool o) {
        if (this.rentalProfit == null && o.rentalProfit == null) {
            return 0;
        }
        else if (this.rentalProfit == null && o.rentalProfit != null) {
            return 1;
        }
        else if (this.rentalProfit != null && o.rentalProfit == null) {
            return -1;
        }
        else {
            // multiply by -1 to inverse the result
            return this.rentalProfit.compareTo(o.rentalProfit) * -1;
        }
    }

    //region Tool-Type classes extend Tool class
    public static class LadderTool extends Tool {

        private Integer stepCount;
        private BigDecimal weightCapacity;

        public LadderTool(int toolID, String type, String powerSource, String subOption, String subType,
                          String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                          BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal weightCapacity, int stepCount) {
            super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                    purchasePrice, rentalPrice, depositPrice);
            this.stepCount = stepCount;
            this.weightCapacity = weightCapacity;
        }

        public Integer getStepCount() {
            return stepCount;
        }

        public BigDecimal getWeightCapacity() {
            return weightCapacity;
        }

        public static class Straight extends LadderTool {

            private Boolean rubberFeet;
            public Straight(int toolID, String type, String powerSource, String subOption, String subType,
                            String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                            BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal weightCapacity,
                            int stepCount, boolean rubberFeet) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, weightCapacity, stepCount);
                this.rubberFeet = rubberFeet;
                createLongDescription();
            }

            public Boolean getRubberFeet() {
                return rubberFeet;
            }

            /*
            If [rubber-feet] IS true, set the substring [rubber-feet] = “ with rubber feet ”.
            If [rubber-feet] IS false or null, set the substring [rubber-feet] = “”.

            Full-Description = [width] + “ in. W. ”+
             [length] + “ in. L. ” +
             [weight] + “ lb. ” +
             [sub-option] + [sub-type] +
             [material] + [weight capacity] + “ lb. capacity “ +
             [step-count] + “ -step ” +
             [rubber feet] + “ by ” + [Manufacturer]

            */
            public void createLongDescription() {
                String feet = "";
                if(rubberFeet != null && rubberFeet == true){feet = " with rubber feet ";}

                String description =
                        this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " ";

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}
                if(this.getWeightCapacity() != null) {description +=
                        this.getWeightCapacity() + " lb. capacity ";}
                if(this.getStepCount() != null) {description +=
                        this.getStepCount() + "-step ";}

                description +=
                        feet +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Step extends LadderTool{

            Boolean pailShelf;
            public Step(int toolID, String type, String powerSource, String subOption, String subType,
                        String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                        BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal weightCapacity, int stepCount, boolean pailShelf) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, weightCapacity, stepCount);
                this.pailShelf = pailShelf;
                createLongDescription();
            }

            public Boolean getPailShelf() {
                return pailShelf;
            }

            /*
            If [pail-shelf] IS true, set the substring [rubber-feet] = “ with pail shelf ”.
            If [pail-shelf] IS false or null, set the substring [rubber-feet] = “”.

            Full-Description = [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb. ” +
            [sub-option] + [sub-type] +
            [material] + [weight capacity] + “ lb. capacity “ +
            [step-count] + “ -step ” +
            [pail-shelf] + “ by ” + [Manufacturer]
            */

            public void createLongDescription() {
                String shelf = "";
                if(pailShelf != null && pailShelf == true){shelf = " with pail shelf ";}

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}
                if(this.getWeightCapacity() != null) {description +=
                        this.getWeightCapacity() + " lb. capacity ";}
                if(this.getStepCount() != null) {description +=
                        this.getStepCount() + "-step ";}

                description +=
                        shelf +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }
    }

    public static class HandTool extends Tool {

        public HandTool(int toolID, String type, String powerSource, String subOption, String subType,
                        String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                        BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice) {
            super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                    purchasePrice, rentalPrice, depositPrice);
        }

        public static class Gun extends HandTool{

            private Integer capacity;
            private Integer gaugeRating;
            public Gun(int toolID, String type, String powerSource, String subOption, String subType,
                       String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                       BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, int capacity, int gaugeRating) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice);
                this.capacity = capacity;
                this.gaugeRating = gaugeRating;
                createLongDescription();
            }

            public Integer getCapacity () {
                return capacity;
            }

            public Integer getGaugeRating(){
                return gaugeRating;
            }

            /*
            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb. ” +
            [sub-option] + [sub-type] + [material] +
            [gauge-rating] + “ G ” +
            [capacity] +
            “ by ” + [Manufacturer]
            */
            private void createLongDescription() {

                String description =
                        this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}
                if(gaugeRating != null) {description +=
                        gaugeRating + " gauge " ;}

                description +=
                        capacity +
                        " by " + this.getManufacturer();
                this.setLongDesc(description);
            }
        }

        public static class Socket extends HandTool{

            private BigDecimal driveSize;
            private BigDecimal saeSize;
            private Boolean deepSocket;
            public Socket(int toolID, String type, String powerSource, String subOption, String subType,
                          String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                          BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal driveSize, BigDecimal saeSize, boolean deepSocket) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice);
                this.driveSize = driveSize;
                this.saeSize = saeSize;
                this.deepSocket = deepSocket;
                createLongDescription();
            }

            public BigDecimal getDriveSize(){return driveSize;}
            public BigDecimal getSaeSize(){return saeSize;}
            public boolean getDeepSocket(){return deepSocket;}

            /*
            If [deep-socket] IS true, set substring [deep-socket] = “ deep-socket “
            If [deep-socket] IS false or null, set substring [deep-socket] = ““

            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb. ” +
            [sub-option] + [sub-type] + [material] +
            [drive-size] + “ in. ” +
            [sae-size] + “ in. ”+
            [deep-socket] +
            “ by ” + [Manufacturer]
            */

            private void createLongDescription() {

                String Socket = "";
                if (deepSocket != null && deepSocket){Socket = " deep-socket ";}

                String description = this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        driveSize + " in. " +
                        saeSize + " in. " +
                        Socket +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Screwdriver extends HandTool{

            private Integer screwSize;
            public Screwdriver(int toolID, String type, String powerSource, String subOption, String subType,
                               String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                               BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, int screwSize) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice);
                this.screwSize = screwSize;
                createLongDescription();
            }
            public Integer getScrewSize (){return screwSize;}


            /*
            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb. ” +
            [sub-option] + [sub-type] + [material] +
            [screw-size] + “ # ” +
            “ by ” + [Manufacturer]

            * */
            private void createLongDescription() {

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        " #" + screwSize +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Hammer extends HandTool{

            private Boolean antiVibration;
            public Hammer(int toolID, String type, String powerSource, String subOption, String subType,
                          String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                          BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, boolean antiVibration) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice);
                this.antiVibration = antiVibration;
                createLongDescription();
            }

            public Boolean getAntiVibration(){return antiVibration;}

            /*
            If [anti-vibration] IS true, set substring [anti-vibration] = “ anti-vibration “
            If [anti-vibration] IS false or null, set substring [anti-vibration] = ““

            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb. ” +
            [sub-option] + [sub-type] + [material] +
            [anti-vibration] +
            “ by ” + [Manufacturer]
            */

            private void createLongDescription() {

                String vibration = "";
                if(antiVibration != null && antiVibration){vibration = " anti-vibration ";}

                String description = this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " ";

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        vibration +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Pliers extends HandTool{

            private Boolean adjustable;
            public Pliers(int toolID, String type, String powerSource, String subOption, String subType,
                         String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                         BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, boolean adjustable) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice);
                this.adjustable = adjustable;
                createLongDescription();
            }

            public Boolean getAdjustable(){return adjustable;}

            /*
            If [adjustable] IS true, set substring [adjustable] = “ adjustable “
            If [adjustable] IS false, set substring [adjustable] = “ non-adjustable “

            Full-Description =
            [width] + “ in. W. ” +
            [length] + “ in. L. ” +
            [weight] + “ lb. ” +
            [sub-option] + [sub-type] + [material] +
            [adjustable] +
            “ by ” + [Manufacturer]
            */

            private void createLongDescription() {

                String adjust = " non-adjustable ";
                if(adjustable != null && adjustable){adjust = " adjustable ";}

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        adjust +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Ratchet extends HandTool{

            private BigDecimal driveSize;
            public Ratchet(int toolID, String type, String powerSource, String subOption, String subType,
                           String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                           BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal driveSize) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice);
                this.driveSize = driveSize;
                createLongDescription();
            }

            public BigDecimal getDriveSize(){return driveSize;}

            /*
            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb. ” +
            [sub-option] + [sub-type] + [material] +
            [drive-size] + “ in. ” +
            “ by ” + [Manufacturer]

            * */
            private void createLongDescription() {

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        driveSize + " in. " +
                        " by " + this.getManufacturer();
                this.setLongDesc(description);
            }
        }

        public static class Wrench extends HandTool {

            private BigDecimal driveSize;

            public Wrench(int toolID, String type, String powerSource, String subOption, String subType,
                          String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                          BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal driveSize) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice);
                this.driveSize = driveSize;
                createLongDescription();
            }

            public BigDecimal getDriveSize() {
                return driveSize;
            }

            /*
            If [drive-size] IS NOT null, set substring [drive-size] = [drive-size] + “ in.“
            If [drive-size] IS null, set substring [drive-size] = ““

            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb. ” +
            [sub-option] + [sub-type] + [material] +
            [drive-size] + “ in. ” +
            “ by ” + [Manufacturer]
            */

            private void createLongDescription() {
                String drive = "";
                if (driveSize != null && driveSize.compareTo(BigDecimal.ZERO)>0){ drive = " " + driveSize + " in. "; }

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        drive +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }
    }

    public static class GardenTool extends Tool {

        private String handleMaterial;

        public GardenTool(int toolID, String type, String powerSource, String subOption, String subType,
                          String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                          BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, String handleMaterial) {
            super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                    purchasePrice, rentalPrice, depositPrice);
            this.handleMaterial = handleMaterial;
        }

        public String getHandleMaterial(){return handleMaterial;}

        public static class Wheelbarrow extends GardenTool{

            private String binMaterial;
            private BigDecimal binVolume;
            private Integer wheelCount;
            public Wheelbarrow(int toolID, String type, String powerSource, String subOption, String subType,
                               String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                               BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, String handleMaterial, String binMaterial, BigDecimal binVolume, int wheelCount) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, handleMaterial);
                this.binMaterial = binMaterial;
                this.binVolume = binVolume;
                this.wheelCount = wheelCount;
                createLongDescription();
            }

            public String getBinMaterial(){return binMaterial;}
            public BigDecimal getBinVolume(){return binVolume;}
            public Integer getWheelCount(){return wheelCount;}

            /*
            If [bin-volume] IS NOT null, set substring [bin-volume] = [bin-volume] + “ cu ft.“
            If [bin-volume] IS null, set substring [blade-material] = ““

            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ”  +
            [weight] + “ lb.” +
            [sub-option] + [sub-type] + [material] +
            [handle-material] + “ handle ” +
            [wheel-count] + “ wheeled” +
            [bin-material] + [bin-volume] +
            “ by ” + [Manufacturer]

            */
            private void createLongDescription() {
                String volume = "";
                if (binVolume != null && binVolume.compareTo(BigDecimal.ZERO)>0){volume = binVolume + " cu. ft. ";}

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        this.getHandleMaterial() + " handle " +
                        wheelCount + " wheeled " +
                        binMaterial +
                        volume +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Digging extends GardenTool{

            private BigDecimal bladeWidth;
            private BigDecimal bladeLength;
            public Digging(int toolID, String type, String powerSource, String subOption, String subType,
                           String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                           BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, String handleMaterial, BigDecimal bladeWidth, BigDecimal bladeLength) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, handleMaterial);
                this.bladeLength = bladeLength;
                this.bladeWidth = bladeWidth;
                createLongDescription();
            }

            public BigDecimal getBladeWidth(){return bladeWidth;}
            public BigDecimal getBladeLength(){return bladeLength;}

            /*
            If [blade-width] IS NOT null, set substring [blade-width] = [blade-width] + “in.“
            If [blade-width] IS null, set substring [blade-width] = ““

            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb.” +
            [sub-option] + [sub-type] + [material] +
            [handle-material] + “ handle ” +
            [blade-width] +
            [blade-length] + “ in.” +
            “ by ” + [Manufacturer]
            */

            private void createLongDescription() {
                String bWidth = "";
                if (bladeWidth != null && bladeWidth.compareTo(BigDecimal.ZERO)>0){bWidth = bladeWidth + " in. ";}

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        this.getHandleMaterial() + " handle " +
                        bWidth +
                        bladeLength + " in. " +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Prunning extends GardenTool{

            private String bladeMaterial = "";
            private BigDecimal bladeLength;

            public Prunning(int toolID, String type, String powerSource, String subOption, String subType,
                            String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                            BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, String handleMaterial, String bladeMaterial, BigDecimal bladeLength) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, handleMaterial);
                this.bladeLength = bladeLength;
                this.bladeMaterial = bladeMaterial;
                createLongDescription();
            }

            public String getBladeMaterial(){return bladeMaterial;}
            public BigDecimal getBladeLength(){return bladeLength;}

            /*
            If [blade-material] IS NOT null, set substring [blade-material] = [blade-material] + “ blade “
            If [blade-material] IS null, set substring [blade-material] = ““

            Full-Description =
            [width] + “ in. W. ” +
            [length] + “ in. L. ”  +
            [weight] + “ lb.” +
            [sub-option] + [sub-type] + [material] +
            [handle-material] + “ handle ” +
            [blade-material] +
            [blade-length] + “ in. ” +
            “ by ” + [Manufacturer]
            */
            private void createLongDescription() {

                String tempBladeMaterial = (this.bladeMaterial != null && !this.bladeMaterial.isEmpty()) ?
                        this.bladeMaterial : "";

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        this.getHandleMaterial() + " handle " +
                        tempBladeMaterial + " " +
                        bladeLength + " in. " +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Striking extends GardenTool{

            private BigDecimal headWeight;
            public Striking(int toolID, String type, String powerSource, String subOption, String subType,
                            String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                            BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, String handleMaterial, BigDecimal headWeight) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, handleMaterial);
                this.headWeight = headWeight;
                createLongDescription();
            }

            public BigDecimal getHeadWeight(){return headWeight;}

            /*
            Full-Description =
            [width] + “ in. W.”+
            [length] + “ in. L.”  +
            [weight] + “ lb.” +
            [sub-option] + [sub-type] + [material] +
            [handle-material] + “ handle” +
            [head-weight] + “ lb. head weight ” +
            “ by ” + [Manufacturer]

            */
            private void createLongDescription() {

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        this.getHandleMaterial() + " handle " +
                        headWeight + " lb. "+
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Rake extends GardenTool{

            private Integer tineCount;
            public Rake(int toolID, String type, String powerSource, String subOption, String subType,
                        String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                        BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, String handleMaterial, int tineCount) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, handleMaterial);
                this.tineCount = tineCount;
                createLongDescription();
            }

            public Integer getTineCount(){return tineCount;}

            /*
            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb.” +
            [sub-option] + [sub-type] + [material] +
            [handle-material] + “ handle ” +
            [tine-count] + “ tine” +
            “ by ” + [Manufacturer]

            */
            private void createLongDescription() {

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        this.getHandleMaterial() + " handle " +
                        tineCount + " tine "+
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }
    }

    public static class PowerTool extends Tool {

        private BigDecimal voltRating;
        private BigDecimal ampRating;
        private BigDecimal minRpmRating;
        private BigDecimal maxRpmRating;
        List<Accessory> accList = null;

        public PowerTool(int toolID, String type, String powerSource, String subOption, String subType,
                         String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                         BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal voltRating,
                         BigDecimal ampRating, BigDecimal minRpmRating, BigDecimal maxRpmRating, List<Accessory> accList) {
            super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                    purchasePrice, rentalPrice, depositPrice);
            this.ampRating = ampRating;
            this.maxRpmRating = maxRpmRating;
            this.minRpmRating = minRpmRating;
            this.voltRating = voltRating;
            this.accList = accList;
        }
//        public PowerTool(int toolID, String powerSource, String subOption, String subType, BigDecimal rentalPrice, BigDecimal depositPrice, float voltRating, float ampRating, float minRpmRating, float maxRpmRating, String batteryType) {
//            super(toolID, powerSource, subOption, subType, rentalPrice, depositPrice);
//            this.ampRating = ampRating;
//            this.maxRpmRating = maxRpmRating;
//            this.minRpmRating = minRpmRating;
//            this.voltRating = voltRating;
//        }

        public BigDecimal getVoltRating(){return voltRating;}
        public BigDecimal getAmpRating(){return ampRating;}
        public BigDecimal getMinRpmRating(){return minRpmRating;}
        public BigDecimal getMaxRpmRating(){return maxRpmRating;}

        public static class Generator extends PowerTool{

            private BigDecimal powerRating;
            public Generator(int toolID, String type, String powerSource, String subOption, String subType,
                             String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                             BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal voltRating,
                             BigDecimal ampRating, BigDecimal minRpmRating, BigDecimal maxRpmRating, BigDecimal powerRating, List<Accessory> accList) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, voltRating, ampRating, minRpmRating, maxRpmRating, accList);
                this.powerRating = powerRating;
                createLongDescription();
            }

            public BigDecimal getPowerRating(){return powerRating;}
            /*
            Full-Description =
            [width] + “ in. W. ” +
            [length] + “ in. L. ” +
            [weight] + “ lb.” +
            [power-source] + [sub-option] + [sub-type] +
            [volt] + “ Volt” +
            [Amp] + ” Amp” +
            [min-rpm-rating][max-rpm-rating] + “ RPM” +
            [material] +
            [power-rating] + “Watt” +
            “ by ” + [Manufacturer]
            */

            private void createLongDescription() {
                BigDecimal minRPM = this.getMinRpmRating();
                BigDecimal maxRPM = this.getMaxRpmRating();
                String Rpm = minRPM + " RPM ";
                if(maxRPM != null && maxRPM.compareTo(BigDecimal.ZERO)>0){Rpm = minRPM + "/" + maxRPM + " RPM ";}

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getPowerSource() + " " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " +
                        this.getVoltRating() + " Volt " +
                        this.getAmpRating() + " Amp " +
                        Rpm ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        powerRating + " Watts " +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Saw extends PowerTool{
            private BigDecimal bladeSize;
            public Saw(int toolID, String type, String powerSource, String subOption, String subType,
                       String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                       BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal voltRating,
                       BigDecimal ampRating, BigDecimal minRpmRating, BigDecimal maxRpmRating, BigDecimal bladeSize, List<Accessory> accList) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, voltRating, ampRating, minRpmRating, maxRpmRating, accList);
                this.bladeSize = bladeSize;
                createLongDescription();
            }

            public BigDecimal getBladeSize(){return bladeSize;}

            /*
            Full-Description =
            [width] + “ in. W. ” +
            [length] + “ in. L. ” +
            [weight] + “ lb.” +
            [power-source] + [sub-option] + [sub-type] +
            [volt] + “ Volt” +
            [Amp] + ” Amp” +
            [min-rpm-rating][max-rpm-rating] + “ RPM” +
            [material] +
            [blade-size] + “ in.” +
            “ by ” + [Manufacturer]

            * */
            private void createLongDescription() {
                BigDecimal minRPM = this.getMinRpmRating();
                BigDecimal maxRPM = this.getMaxRpmRating();
                String Rpm = minRPM + " RPM ";
                if(maxRPM != null && maxRPM.compareTo(BigDecimal.ZERO)>0){Rpm = minRPM + "/" + maxRPM + " RPM ";}

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getPowerSource() + " " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " +
                        this.getVoltRating() + " Volt " +
                        this.getAmpRating() + " Amp " +
                        Rpm ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        bladeSize + " in. " +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Sander extends PowerTool{

            private Boolean dustBag;
            public Sander(int toolID, String type, String powerSource, String subOption, String subType,
                          String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                          BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal voltRating,
                          BigDecimal ampRating, BigDecimal minRpmRating, BigDecimal maxRpmRating, Boolean dustBag, List<Accessory> accList) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, voltRating, ampRating, minRpmRating, maxRpmRating, accList);
                this.dustBag = dustBag;
                createLongDescription();

            }

            public Boolean getDustBag(){return dustBag;}

            /*
            If [dust-bag] IS true, set substring [dust-bag] = “ dust-bag“.
            If [dust-bag] IS false, set substring [dust-bag] = “ no dust-bag“.

            Full-Description =
            [width] + “ in. W. ” +
            [length] + “ in. L. ” +
            [weight] + “ lb.” +
            [power-source] + [sub-option] + [sub-type] +
            [volt] + “ Volt” +
            [Amp] + ” Amp” +
            [min-rpm-rating][max-rpm-rating] + “ RPM” +
            [material] +
            [dust-bag] +
            “ by ” + [Manufacturer]


            * */
            private void createLongDescription() {
                BigDecimal minRPM = this.getMinRpmRating();
                BigDecimal maxRPM = this.getMaxRpmRating();
                String Rpm = minRPM + " RPM ";
                if(maxRPM != null && maxRPM.compareTo(BigDecimal.ZERO)>0){Rpm = minRPM + "/" + maxRPM + " RPM ";}

                String bag = " no dust-bag ";
                if(dustBag != null && dustBag){bag = " dust-bag ";}

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getPowerSource() + " " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " +
                        this.getVoltRating() + " Volt " +
                        this.getAmpRating() + " Amp " +
                        Rpm ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        bag +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class AirCompressor extends PowerTool{

            private BigDecimal tankSize;
            private BigDecimal pressureRating;
            public AirCompressor(int toolID, String type, String powerSource, String subOption, String subType,
                                 String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                                 BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal voltRating,
                                 BigDecimal ampRating, BigDecimal minRpmRating, BigDecimal maxRpmRating, BigDecimal tankSize, BigDecimal pressureGauge, List<Accessory> accList) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, voltRating, ampRating, minRpmRating, maxRpmRating, accList);
                this.tankSize = tankSize;
                this.pressureRating = pressureRating;
                createLongDescription();
            }

            public BigDecimal getTankSize(){return tankSize;}
            public BigDecimal getPressureRating(){return  pressureRating;}

            /*
            If [pressure-rating] IS NOT null, set substring [pressure-rating] = [pressure-rating] + “ psi”
            If [pressure-rating] IS null, set substring [pressure-rating] = ““.

            Full-Description =
            [width] + “ in. W. ” +
            [length] + “ in. L. ” +
            [weight] + “ lb.” +
            [power-source] + [sub-option] + [sub-type] +
            [volt] + “ Volt” +
            [Amp] + ” Amp” +
            [min-rpm-rating][max-rpm-rating] + “ RPM” +
            [material] +
            [tank-size] + “ gal” +
            [pressure-rating] +
            “ by ” + [Manufacturer]


            * */
            private void createLongDescription() {
                BigDecimal minRPM = this.getMinRpmRating();
                BigDecimal maxRPM = this.getMaxRpmRating();
                String Rpm = minRPM + " RPM ";
                if(maxRPM != null && maxRPM.compareTo(BigDecimal.ZERO)>0){Rpm = minRPM + "/" + maxRPM + " RPM ";}

                String pressure = "";
                if(pressureRating != null && pressureRating.compareTo(BigDecimal.ZERO)>0){pressure = pressureRating + " psi ";}

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getPowerSource() + " " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " +
                        this.getVoltRating() + " Volt " +
                        this.getAmpRating() + " Amp " +
                        Rpm ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        tankSize + " gal" +
                        pressure +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static class Drill extends PowerTool{

            private Boolean adjustableClutch = false;
            private BigDecimal minTorqueRating;
            private BigDecimal maxTorqueRating;
            public Drill(int toolID, String type, String powerSource, String subOption, String subType,
                         String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                         BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal voltRating,
                         BigDecimal ampRating, BigDecimal minRpmRating, BigDecimal maxRpmRating, Boolean adjustableClutch, BigDecimal minTorqueRating, BigDecimal maxTorqueRating, List<Accessory> accList) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, voltRating, ampRating, minRpmRating, maxRpmRating, accList);
                this.adjustableClutch = adjustableClutch;
                this.minTorqueRating = minTorqueRating;
                this.maxTorqueRating = maxTorqueRating;
                createLongDescription();
            }

            public Boolean getAdjustableClutch(){return adjustableClutch;}
            public BigDecimal getMinTorqueRating(){return minTorqueRating;}
            public BigDecimal getMaxTorqueRating(){return maxTorqueRating;}

            /*
            If [adjustable-clutch] IS true, set substring [adjustable-clutch] = “ adjustable clutch”
            If [adjustable-clutch] IS false or null, set substring [adjustable-clutch] = ““.

            If [max-torque-rating] IS NOT null, set substring  = “/” + [max-torque-rating]
            If [max-torque-rating] IS null, set substring [max-torque-rating] = ““.

            Full-Description =
            [width] + “ in. W. ”+
            [length] + “ in. L. ” +
            [weight] + “ lb. ” +
            [power-source] + [sub-option] + [sub-type] +
            [volt] + “ Volt” +
            [Amp] + ” Amp” +
            [min-rpm-rating][max-rpm-rating] + “ RPM” +
            [material] +
            [adjustable-clutch] +
            [min-torque-rating][max-torque-rating] + “ ft-lb” +
            “ by ” + [Manufacturer]
            */
            private void createLongDescription() {
                BigDecimal minRPM = this.getMinRpmRating();
                BigDecimal maxRPM = this.getMaxRpmRating();
                String Rpm = minRPM + " RPM ";
                if(maxRPM != null && maxRPM.compareTo(BigDecimal.ZERO)>0){Rpm = minRPM + "/" + maxRPM + " RPM ";}

                String clutch = "";
                if(adjustableClutch != null && adjustableClutch){clutch = " adjustable clutch ";}

                String torque = "";
                if(maxTorqueRating != null && maxTorqueRating.compareTo(BigDecimal.ZERO)>0) {
                    torque = minTorqueRating + "/" + maxTorqueRating + " ft-lb ";
                }
                else{
                    torque = minTorqueRating + " ft-lb ";
                }

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getPowerSource() + " " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " +
                        this.getVoltRating() + " Volt " +
                        this.getAmpRating() + " Amp " +
                        Rpm + " " ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        clutch +
                        torque +
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }

        public static  class Mixer extends PowerTool{

            private BigDecimal motorRating;
            private BigDecimal drumSize;

            public Mixer(int toolID, String type, String powerSource, String subOption, String subType,
                         String material, BigDecimal length, BigDecimal width, BigDecimal weight, String manufacturer,
                         BigDecimal purchasePrice, BigDecimal rentalPrice, BigDecimal depositPrice, BigDecimal voltRating,
                         BigDecimal ampRating, BigDecimal minRpmRating, BigDecimal maxRpmRating, BigDecimal motorRating, BigDecimal drumSize, List<Accessory> accList) {
                super(toolID, type, powerSource, subOption, subType, material, length, width, weight, manufacturer,
                        purchasePrice, rentalPrice, depositPrice, voltRating, ampRating, minRpmRating, maxRpmRating, accList);
                this.motorRating = motorRating;
                this.drumSize = drumSize;
                createLongDescription();
            }

            public BigDecimal getDrumSize(){return drumSize;}
            public BigDecimal getMotorRating(){return motorRating;}

            /*
            Full-Description =
            [width] + “ in. W. ” +
            [length] + “ in. L. ” +
            [weight] + “ lb.” +
            [power-source] + [sub-option] + [sub-type] +
            [volt] + “ Volt” +
            [Amp] + ” Amp” +
            [min-rpm-rating][max-rpm-rating] + “ RPM” +
            [material] +
            [motor-rating] + “ HP” +
            [drum-size] + “ cu.ft.” +
            “ by ” + [Manufacturer]
            */

            private void createLongDescription() {
                BigDecimal minRPM = this.getMinRpmRating();
                BigDecimal maxRPM = this.getMaxRpmRating();
                String Rpm = minRPM + " RPM ";
                if(maxRPM != null && maxRPM.compareTo(BigDecimal.ZERO)>0){Rpm = minRPM + "/" + maxRPM + " RPM ";}

                String description =  this.getWidth() + " in. W. " +
                        this.getLength() + " in. L. " +
                        this.getWeight() + " lb. " +
                        this.getPowerSource() + " " +
                        this.getSubOption() + " " +
                        this.getSubType() + " " +
                        this.getVoltRating() + " Volt " +
                        this.getAmpRating() + " Amp " +
                        Rpm ;

                if(this.getMaterial() != null) {description +=
                        this.getMaterial() + " ";}

                description +=
                        motorRating + " HP ";

                if(drumSize != null) {description +=
                        drumSize + " cu.ft. ";}

                description +=
                        " by " + this.getManufacturer();

                this.setLongDesc(description);
            }
        }
    }
    //endregion
}
