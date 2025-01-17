''' Conflict resolution based on the Modified Voltage Potential algorithm. '''
import numpy as np
from bluesky import stack
import bluesky as bs
from bluesky.core import Entity

from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import translate, scale
from shapely.ops import nearest_points

import numpy as np
from math import *

class VO(Entity):
    """ 
    Conflict Resolution - Modified Voltage Potential

    Taken from BlueSky library, but made "adjustable" as a separate script here
    """
    def __init__(self):
        super().__init__()
        # [-] switch to limit resolution to the horizontal direction
        self.swresohoriz = True
        # [-] switch to use only speed resolutions (works with swresohoriz = True)
        self.swresospd = False
        # [-] switch to use only heading resolutions (works with swresohoriz = True)
        self.swresohdg = False
        # [-] switch to limit resolution to the vertical direction
        self.swresovert = False

        self.swprio = False  # switch priority on/off
        self.priocode = ''  # select priority mode
        self.resopairs = set()  # Resolved conflicts that are still before CPA

        # Resolution factors:
        # set < 1 to maneuver only a fraction of the resolution
        # set > 1 to add a margin to separation values
        self.resofach = bs.settings.asas_marh
        self.resofacv = bs.settings.asas_marv
        
        # Switches to guarantee last reso zone commands keep valid if cd zone changes
        self.resodhrelative = True # Size of resolution zone dh, vertically, set relative to CD zone
        self.resorrelative  = True # Size of resolution zone r, vertically, set relative to CD zone

        with self.settrafarrays():
            self.resooffac = np.array([], dtype=bool)
            self.noresoac = np.array([], dtype=bool)
            # whether the autopilot follows ASAS or not
            self.active = np.array([], dtype=bool)
            self.trk = np.array([])  # heading provided by the ASAS [deg]
            self.tas = np.array([])  # speed provided by the ASAS (eas) [m/s]
            self.alt = np.array([])  # alt provided by the ASAS [m]
            self.vs = np.array([])  # vspeed provided by the ASAS [m/s]

    def setprio(self, flag=None, priocode=''):
        '''Set the prio switch and the type of prio '''
        if flag is None:
            return True, "PRIORULES [ON/OFF] [PRIOCODE]" + \
                            "\nAvailable priority codes: " + \
                            "\n     FF1:  Free Flight Primary (No Prio) " + \
                            "\n     FF2:  Free Flight Secondary (Cruising has priority)" + \
                            "\n     FF3:  Free Flight Tertiary (Climbing/descending has priority)" + \
                            "\n     LAY1: Layers Primary (Cruising has priority + horizontal resolutions)" + \
                            "\n     LAY2: Layers Secondary (Climbing/descending has priority + horizontal resolutions)" + \
                            "\nPriority is currently " + ("ON" if self.swprio else "OFF") + \
                            "\nPriority code is currently: " + \
                str(self.priocode)
        options = ["FF1", "FF2", "FF3", "LAY1", "LAY2"]
        if priocode not in options:
            return False, "Priority code Not Understood. Available Options: " + str(options)
        return super().setprio(flag, priocode)

    def applyprio(self, dv_vo, dv1, dv2, vs1, vs2):
        ''' Apply the desired priority setting to the resolution '''

        # Primary Free Flight prio rules (no priority)
        if self.priocode == 'FF1':
            # since cooperative, the vertical resolution component can be halved, and then dv_vo can be added
            dv_vo[2] = dv_vo[2] / 2.0
            dv1 = dv1 - dv_vo
            dv2 = dv2 + dv_vo

        # Secondary Free Flight (Cruising aircraft has priority, combined resolutions)
        if self.priocode == 'FF2':
            # since cooperative, the vertical resolution component can be halved, and then dv_vo can be added
            dv_vo[2] = dv_vo[2]/2.0
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 2 solves conflict
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv2 = dv2 + dv_vo
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 1 solves conflict
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv1 = dv1 - dv_vo
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflict
                dv1 = dv1 - dv_vo
                dv2 = dv2 + dv_vo

        # Tertiary Free Flight (Climbing/descending aircraft have priority and crusing solves with horizontal resolutions)
        elif self.priocode == 'FF3':
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 1 solves conflict horizontally
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv_vo[2] = 0.0
                dv1 = dv1 - dv_vo
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 2 solves conflict horizontally
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv_vo[2] = 0.0
                dv2 = dv2 + dv_vo
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflict, combined
                dv_vo[2] = dv_vo[2]/2.0
                dv1 = dv1 - dv_vo
                dv2 = dv2 + dv_vo

        # Primary Layers (Cruising aircraft has priority and clmibing/descending solves. All conflicts solved horizontally)
        elif self.priocode == 'LAY1':
            dv_vo[2] = 0.0
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 2 solves conflict horizontally
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv2 = dv2 + dv_vo
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 1 solves conflict horizontally
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv1 = dv1 - dv_vo
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflict horizontally
                dv1 = dv1 - dv_vo
                dv2 = dv2 + dv_vo

        # Secondary Layers (Climbing/descending aircraft has priority and cruising solves. All conflicts solved horizontally)
        elif self.priocode == 'LAY2':
            dv_vo[2] = 0.0
            # If aircraft 1 is cruising, and aircraft 2 is climbing/descending -> aircraft 1 solves conflict horizontally
            if abs(vs1) < 0.1 and abs(vs2) > 0.1:
                dv1 = dv1 - dv_vo
            # If aircraft 2 is cruising, and aircraft 1 is climbing -> aircraft 2 solves conflict horizontally
            elif abs(vs2) < 0.1 and abs(vs1) > 0.1:
                dv2 = dv2 + dv_vo
            else:  # both are climbing/descending/cruising -> both aircraft solves the conflic horizontally
                dv1 = dv1 - dv_vo
                dv2 = dv2 + dv_vo

        return dv1, dv2

    def resumenav(self, conf, ownship, intruder):
        self.resopairs.update(conf.confpairs)

        delpairs = set()
        changeactive = dict()

        # smallest relative angle between vectors of heading a and b
        def anglediff(a, b):
            d = a - b
            if d > 180:
                return anglediff(a, b + 360)
            elif d < -180:
                return anglediff(a + 360, b)
            else:
                return d

        # Look at all conflicts, also the ones that are solved but CPA is yet to come
        for conflict in self.resopairs:
            idx1, idx2 = bs.traf.id2idx(conflict)
            # If the ownship aircraft is deleted remove its conflict from the list
            if idx1 < 0:
                delpairs.add(conflict)
                continue

            if idx2 >= 0:
                # Distance vector using flat earth approximation
                re = 6371000.
                dist = re * np.array([np.radians(intruder.lon[idx2] - ownship.lon[idx1]) *
                                      np.cos(0.5 * np.radians(intruder.lat[idx2] +
                                                              ownship.lat[idx1])),
                                      np.radians(intruder.lat[idx2] - ownship.lat[idx1])])

                # Relative velocity vector
                vrel = np.array([intruder.gseast[idx2] - ownship.gseast[idx1],
                                 intruder.gsnorth[idx2] - ownship.gsnorth[idx1]])

                # Check if conflict is past CPA
                past_cpa = np.dot(dist, vrel) > 0.0

                rpz = np.max(conf.rpz[[idx1, idx2]])
                # hor_los:
                # Aircraft should continue to resolve until there is no horizontal
                # LOS. This is particularly relevant when vertical resolutions
                # are used.
                hdist = np.linalg.norm(dist)
                hor_los = hdist < rpz

                # Bouncing conflicts:
                # If two aircraft are getting in and out of conflict continously,
                # then they it is a bouncing conflict. ASAS should stay active until
                # the bouncing stops.
                is_bouncing = \
                    abs(anglediff(ownship.trk[idx1], intruder.trk[idx2])) < 30.0 and \
                    hdist < rpz * self.resofach

            # Start recovery for ownship if intruder is deleted, or if past CPA
            # and not in horizontal LOS or a bouncing conflict
            if idx2 >= 0 and (not past_cpa or hor_los or is_bouncing):
                # Enable ASAS for this aircraft
                changeactive[idx1] = True
            else:
                # Switch ASAS off for ownship if there are no other conflicts
                # that this aircraft is involved in.
                changeactive[idx1] = changeactive.get(idx1, False)
                # If conflict is solved, remove it from the resopairs list
                delpairs.add(conflict)

        for idx, active in changeactive.items():
            # Loop a second time: this is to avoid that ASAS resolution is
            # turned off for an aircraft that is involved simultaneously in
            # multiple conflicts, where the first, but not all conflicts are
            # resolved.
            self.active[idx] = active
            if not active:
                # Waypoint recovery after conflict: Find the next active waypoint
                # and send the aircraft to that waypoint.
                iwpid = bs.traf.ap.route[idx].findact(idx)
                if iwpid != -1:  # To avoid problems if there are no waypoints
                    bs.traf.ap.route[idx].direct(
                        idx, bs.traf.ap.route[idx].wpname[iwpid])

        # Remove pairs from the list that are past CPA or have deleted aircraft
        self.resopairs -= delpairs

    def resolve(self, conf, ownship, intruder):
        ''' Resolve all current conflicts '''
        # Initialize an array to store the resolution velocity vector for all A/C
        dv = np.zeros((ownship.ntraf, 3))

        # Initialize an array to store time needed to resolve vertically
        timesolveV = np.ones(ownship.ntraf) * 1e9

        # Call vo function to resolve conflicts-----------------------------------
        for ((ac1, ac2), qdr, dist, tcpa, tLOS) in zip(conf.confpairs, conf.qdr, conf.dist, conf.tcpa, conf.tLOS):
            idx1 = ownship.id.index(ac1)
            idx2 = intruder.id.index(ac2)

            # If A/C indexes are found, then apply vo on this conflict pair
            # Because ADSB is ON, this is done for each aircraft separately
            if idx1 >-1 and idx2 > -1:
                dv_vo, tsolV = self.VO(ownship, intruder, conf, qdr, dist, tLOS, idx1, idx2)
                if tsolV < timesolveV[idx1]:
                    timesolveV[idx1] = tsolV

                # Use priority rules if activated
                if self.swprio:
                    dv[idx1], _ = self.applyprio(dv_vo, dv[idx1], dv[idx2], ownship.vs[idx1], intruder.vs[idx2])
                else:
                    # since cooperative, the vertical resolution component can be halved, and then dv_vo can be added
                    dv_vo[2] = 0.5 * dv_vo[2]
                    dv[idx1] = dv[idx1] - dv_vo

                # Check the noreso aircraft. Nobody avoids noreso aircraft.
                # But noreso aircraft will avoid other aircraft
                if self.noresoac[idx2]:
                    dv[idx1] = dv[idx1] + dv_vo

                # Check the resooff aircraft. These aircraft will not do resolutions.
                if self.resooffac[idx1]:
                    dv[idx1] = 0.0


        # Determine new speed and limit resolution direction for all aicraft-------

        # Resolution vector for all aircraft, cartesian coordinates
        dv = np.transpose(dv)

        # The old speed vector, cartesian coordinates
        v = np.array([ownship.gseast, ownship.gsnorth, ownship.vs])

        # The new speed vector, cartesian coordinates
        newv = v + dv

        # Limit resolution direction if required-----------------------------------

        # Compute new speed vector in polar coordinates based on desired resolution
        if self.swresohoriz: # horizontal resolutions
            if self.swresospd and not self.swresohdg: # SPD only
                newtrack = ownship.trk
                newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
                newvs    = ownship.vs
            elif self.swresohdg and not self.swresospd: # HDG only
                newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) % 360
                newgs    = ownship.gs
                newvs    = ownship.vs
            else: # SPD + HDG
                newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
                newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
                newvs    = ownship.vs
        elif self.swresovert: # vertical resolutions
            newtrack = ownship.trk
            newgs    = ownship.gs
            newvs    = newv[2,:]
        else: # horizontal + vertical
            newtrack = (np.arctan2(newv[0,:],newv[1,:])*180/np.pi) %360
            newgs    = np.sqrt(newv[0,:]**2 + newv[1,:]**2)
            newvs    = newv[2,:]

        # Determine ASAS module commands for all aircraft--------------------------

        # Cap the velocity
        newgscapped = np.maximum(ownship.perf.vmin,np.minimum(ownship.perf.vmax,newgs))

        # Cap the vertical speed
        vscapped = np.maximum(ownship.perf.vsmin,np.minimum(ownship.perf.vsmax,newvs))

        # Calculate if Autopilot selected altitude should be followed. This avoids ASAS from
        # climbing or descending longer than it needs to if the autopilot leveloff
        # altitude also resolves the conflict. Because asasalttemp is calculated using
        # the time to resolve, it may result in climbing or descending more than the selected
        # altitude.
        asasalttemp = vscapped * timesolveV + ownship.alt
        signdvs = np.sign(vscapped - ownship.ap.vs * np.sign(ownship.selalt - ownship.alt))
        signalt = np.sign(asasalttemp - ownship.selalt)
        alt = np.where(np.logical_or(signdvs == 0, signdvs == signalt), asasalttemp, ownship.selalt)

        # To compute asas alt, timesolveV is used. timesolveV is a really big value (1e9)
        # when there is no conflict. Therefore asas alt is only updated when its
        # value is less than the look-ahead time, because for those aircraft are in conflict
        altCondition = np.logical_and(timesolveV<conf.dtlookahead, np.abs(dv[2,:])>0.0)
        alt[altCondition] = asasalttemp[altCondition]

        # If resolutions are limited in the horizontal direction, then asasalt should
        # be equal to auto pilot alt (aalt). This is to prevent a new asasalt being computed
        # using the auto pilot vertical speed (ownship.avs) using the code in line 106 (asasalttemp) when only
        # horizontal resolutions are allowed.
        alt = alt * (1 - self.swresohoriz) + ownship.selalt * self.swresohoriz

        self.resumenav(conf, ownship, intruder)

        return newtrack, newgscapped, vscapped, alt, self.resopairs

    def get_cc_tp(self, ownship_position, intruder_position, rpz):
        dx = intruder_position.x - ownship_position.x
        dy = intruder_position.y - ownship_position.y

        # print("Body frame: ", dx, dy)

        d = sqrt(dx**2 + dy**2)

        if(d > rpz):
            theta = atan2(dy, dx)
            beta = asin(rpz/d)
            side = sqrt(d**2 - rpz**2)

            tp_1_x = ownship_position.x + side * cos(theta - beta)
            tp_1_y = ownship_position.y + side * sin(theta - beta)
            tp_2_x = ownship_position.x + side * cos(theta + beta)
            tp_2_y = ownship_position.y + side * sin(theta + beta)

            return Point(tp_1_x, tp_1_y), Point(tp_2_x, tp_2_y)
        
        else:
            return None, None
        
    def VO(self, ownship, intruder, conf, qdr, dist, tLOS, idx1, idx2):
        
        ##ownship_position, ownship_gs, ownship_trk,
        ##   intruder_position, intruder_gs, intruder_trk,
        ##   rpz, method = 0):

        rpz = np.max(conf.rpz[[idx1, idx2]] * self.resofach)
        hpz = np.max(conf.hpz[[idx1, idx2]] * self.resofach)
        dtlook = conf.dtlookahead[idx1]
    
        # Convert qdr from degrees to radians
        qdr = np.radians(qdr)

        # Relative position vector between id1 and id2
        drel = np.array([np.sin(qdr) * dist, \
                        np.cos(qdr) * dist, \
                        intruder.alt[idx2] - ownship.alt[idx1]])

        ownship_position = Point(0, 0)

        intruder_position = Point(drel[1], drel[0])
        

        tp_1, tp_2 = self.get_cc_tp(ownship_position, intruder_position, rpz)
        
        ownship_velocity = Point(ownship.gsnorth[idx1], ownship.gseast[idx1])
        intruder_velocity = Point(intruder.gsnorth[idx2], intruder.gseast[idx2])
        
        method = 0

        if((tp_1 != None) & (tp_2 != None)):
            vo_0 = translate(ownship_position, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
            vo_1 = translate(tp_1, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
            vo_2 = translate(tp_2, xoff = intruder_velocity.x, yoff = intruder_velocity.y)

            vo_line_1 = LineString([vo_0, vo_1])
            vo_line_2 = LineString([vo_0, vo_2])

            # method = 0: opt, 1: spd change, 2: hdg change
            if(method == 0):
                cp_1 = nearest_points(vo_line_1, ownship_velocity)[0]
                cp_2 = nearest_points(vo_line_2, ownship_velocity)[0]

                cp_1_dist = cp_1.distance(ownship_velocity)
                cp_2_dist = cp_2.distance(ownship_velocity)

                if(cp_1_dist <= cp_2_dist):
                    cp = cp_1
                else:
                    cp = cp_2

            dv1 = ownship_velocity.y - cp.y
            dv2 = ownship_velocity.x - cp.x

        else:
            dv1 = 0
            dv2 = 0

        ##### ----------------------------------------
        # Compute the  vertical intrusion
        # Amount of vertical intrusion dependent on vertical relative velocity
        vrel = [ownship_velocity.y - intruder_velocity.y, ownship_velocity.x - intruder_velocity.x, 0]

        iV = hpz if abs(vrel[2]) > 0.0 else hpz - abs(drel[2])

        # Get the time to solve the conflict vertically - tsolveV
        tsolV = abs(drel[2] / vrel[2]) if abs(vrel[2]) > 0.0 else tLOS

        # If the time to solve the conflict vertically is longer than the look-ahead time,
        # because the the relative vertical speed is very small, then solve the intrusion
        # within tinconf
        if tsolV > dtlook:
            tsolV = tLOS
            iV    = hpz

        # Compute the resolution velocity vector in the vertical direction
        # The direction of the vertical resolution is such that the aircraft with
        # higher climb/decent rate reduces their climb/decent rate
        dv3 = 0
        
        dv = np.array([dv1, dv2, dv3])

        return dv, tsolV
    
        """Modified Voltage Potential (vo) resolution method"""
        # Preliminary calculations-------------------------------------------------
        # Determine largest RPZ and HPZ of the conflict pair, use lookahead of ownship
        rpz_m = np.max(conf.rpz[[idx1, idx2]] * self.resofach)
        hpz_m = np.max(conf.hpz[[idx1, idx2]] * self.resofacv)
        dtlook = conf.dtlookahead[idx1]
        # Convert qdr from degrees to radians
        qdr = np.radians(qdr)

        # Relative position vector between id1 and id2
        drel = np.array([np.sin(qdr) * dist, \
                        np.cos(qdr) * dist, \
                        intruder.alt[idx2] - ownship.alt[idx1]])

        # Write velocities as vectors and find relative velocity vector
        v1 = np.array([ownship.gseast[idx1], ownship.gsnorth[idx1], ownship.vs[idx1]])
        v2 = np.array([intruder.gseast[idx2], intruder.gsnorth[idx2], intruder.vs[idx2]])
        vrel = v2 - v1


        # Horizontal resolution----------------------------------------------------

        # Find horizontal distance at the tcpa (min horizontal distance)
        dcpa  = drel + vrel*tcpa
        dabsH = np.sqrt(dcpa[0] * dcpa[0] + dcpa[1] * dcpa[1])

        # Compute horizontal intrusion
        iH = rpz_m - dabsH

        # Exception handlers for head-on conflicts
        # This is done to prevent division by zero in the next step
        threshold = 0.001
        
        if dabsH <= threshold:
            dabsH = threshold
            dcpa[0] = drel[1] / dist * dabsH
            dcpa[1] = -drel[0] / dist * dabsH

        # If intruder is outside the ownship PZ, then apply extra factor
        # to make sure that resolution does not graze IPZ
        if rpz_m < dist and dabsH < dist:
            # Compute the resolution velocity vector in horizontal direction.
            # abs(tcpa) because it bcomes negative during intrusion.
            erratum = np.cos(np.arcsin(rpz_m / dist)-np.arcsin(dabsH / dist))
            dv1 = ((rpz_m / erratum - dabsH) * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = ((rpz_m / erratum - dabsH) * dcpa[1]) / (abs(tcpa) * dabsH)
        else:
            dv1 = (iH * dcpa[0]) / (abs(tcpa) * dabsH)
            dv2 = (iH * dcpa[1]) / (abs(tcpa) * dabsH)

        # Vertical resolution------------------------------------------------------

        # Compute the  vertical intrusion
        # Amount of vertical intrusion dependent on vertical relative velocity
        iV = hpz_m if abs(vrel[2]) > 0.0 else hpz_m - abs(drel[2])

        # Get the time to solve the conflict vertically - tsolveV
        tsolV = abs(drel[2] / vrel[2]) if abs(vrel[2]) > 0.0 else tLOS

        # If the time to solve the conflict vertically is longer than the look-ahead time,
        # because the the relative vertical speed is very small, then solve the intrusion
        # within tinconf
        if tsolV > dtlook:
            tsolV = tLOS
            iV    = hpz_m

        # Compute the resolution velocity vector in the vertical direction
        # The direction of the vertical resolution is such that the aircraft with
        # higher climb/decent rate reduces their climb/decent rate
        dv3 = np.where(abs(vrel[2]) > 0.0, (iV / tsolV) * (-vrel[2] / abs(vrel[2])), (iV / tsolV))

        # It is necessary to cap dv3 to prevent that a vertical conflict
        # is solved in 1 timestep, leading to a vertical separation that is too
        # high (high vs assumed in traf). If vertical dynamics are included to
        # aircraft  model in traffic.py, the below three lines should be deleted.
        #    mindv3 = -400*fpm# ~ 2.016 [m/s]
        #    maxdv3 = 400*fpm
        #    dv3 = np.maximum(mindv3,np.minimum(maxdv3,dv3))


        # Combine resolutions------------------------------------------------------

        # combine the dv components
        dv = np.array([dv1, dv2, dv3])

        return dv, tsolV
