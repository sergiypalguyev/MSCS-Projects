package edu.gatech.dbclass.web;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import edu.gatech.dbclass.t4r.Database;
import edu.gatech.dbclass.t4r.JSONConverter;
import edu.gatech.dbclass.t4r.Status;
import edu.gatech.dbclass.t4r.Tool;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Type;

public class AddNewToolServlet extends HttpServlet {

    @Override
    protected void doOptions(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_OK);
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        Options.setAccessControlHeaders(response);
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json");
        PrintWriter pw = response.getWriter();

        try {
            Status status = new Status();
            StringBuffer sb = Options.getRequestData(request);
            JsonParser jp = new JsonParser();
            JsonObject jo = jp.parse(sb.toString()).getAsJsonObject();
            String subType = jo.get("subType").getAsString();
            Type t = null;

            switch (subType.toUpperCase()) {
                //-----Ladder Tools-----
                case "STRAIGHT":
                    t = Tool.LadderTool.Straight.class;
                    break;

                case "STEP":
                    t = Tool.LadderTool.Step.class;
                    break;

                //-----Hand Tools-----
                case "SCREWDRIVER":
                    t = Tool.HandTool.Screwdriver.class;
                    break;

                case "SOCKET":
                    t = Tool.HandTool.Socket.class;
                    break;

                case "RATCHET":
                    t = Tool.HandTool.Ratchet.class;
                    break;

                case "WRENCH":
                    t = Tool.HandTool.Wrench.class;
                    break;

                case "PLIERS":
                    t = Tool.HandTool.Pliers.class;
                    break;

                case "GUN":
                    t = Tool.HandTool.Gun.class;
                    break;

                case "HAMMER":
                    t = Tool.HandTool.Hammer.class;
                    break;

                //-----Garden Tools-----
                case "DIGGER":
                    t = Tool.GardenTool.Digging.class;
                    break;

                case "PRUNER":
                    t = Tool.GardenTool.Prunning.class;
                    break;

                case "RAKES":
                    t = Tool.GardenTool.Rake.class;
                    break;

                case "WHEELBARROWS":
                    t = Tool.GardenTool.Wheelbarrow.class;
                    break;

                case "STRIKING":
                    t = Tool.GardenTool.Striking.class;
                    break;

                //-----Power Tools-----
                case "DRILL":
                    t = Tool.PowerTool.Drill.class;
                    break;

                case "SAW":
                    t = Tool.PowerTool.Saw.class;
                    break;

                case "SANDER":
                    t = Tool.PowerTool.Sander.class;
                    break;

                case "AIR-COMPRESSOR":
                    t = Tool.PowerTool.AirCompressor.class;
                    break;

                case "MIXER":
                    t = Tool.PowerTool.Mixer.class;
                    break;

                case "GENERATOR":
                    t = Tool.PowerTool.Generator.class;
                    break;

                default:
                    break;
            }


            if (t == null) {
                status.errorCode = -2;
                status.errorMessage = "Unable to parse JSON object.";
                pw.println(JSONConverter.getInstance().getGson().toJson(status));
                return;
            }

            Tool tool = JSONConverter.getInstance().getGson().fromJson(jo, t);
            status = Database.getInstance().addNewTool(tool);

            if (status.errorCode >= 0) {
                response.setStatus(HttpServletResponse.SC_OK);
            }

            pw.println(JSONConverter.getInstance().getGson().toJson(status));

        } catch (Exception e) {
            // crash and burn
            throw new IOException(e.getMessage());
        }
    }
}
