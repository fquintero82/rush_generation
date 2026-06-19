from numba import cuda
import numpy as np


def get_name_outputs():
  d={'STATE_RUNOFF':0,
        'STATE_PRECIP':1,
        'STATE_ACTUALET':2,
        'STATE_STATIC':3,
        'STATE_SURFACE':4,
        'STATE_TOPLAYER':5,
        'STATE_BOTTOMLAYER':6,
        'STATE_SURFACE_FLOW':7,
        'STATE_INTERFLOW':8,
        'STATE_TDFLOW':9,
        'STATE_BASEFLOW':10}
  return d

def get_n_outputs():
  return len(get_name_outputs())


@cuda.jit
def hlm_kernel(timesteps, nhills,
    doy_init,
    model_output,
    precipitation,
    evapotranspiration,
    temperature,
    area_hillslope,
    static, #state
    surface, #state
    toplayer, #state
    bottomlayer,#state
    swe, #state
    bedrock, #param,meters
    toplayer_porosity, #param, average porosity from 0 to 15cm,m/m
    bottomlayer_porosity,#param,average porosity from 15 to 200cm,m/m
    ksat_toplayer, #param, infiltration when toplayer is saturated, mm/h
    ksat_bottomlayer,#param, percolation when bottomlayer is saturated,mm/h
    td_factor,#param, 0 to 1 degree of tile drain
    parameters):

  



  CF_MMHR_M_MIN = (float)(1./1000.)*(1/60.) #factor .converts [mm/hr] to [m/min]
  CF_MELTFACTOR= (float)((1/(24*60.0)) *(1/1000.0)) # mm/day/degree to m/min/degree
  #CF_ET = (float)((1e-3 / (30.0*24.0*60.0))) #mm/month to m/min
  CF_MMDAY_M_MIN = (float)(1./1000.)*(1/60)*(1/24) #mm/day to m/min
  CF_METER_TO_MM = float(1000)
  CF_DAYS_TO_MINUTES = float(24 * 60)
  DT = float(60) #minutes
  
  #

  
  #index
  
  tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
  ty = cuda.blockIdx.x  # Similarly, this is the unique block ID within the 1D grid
  block_size = cuda.blockDim.x  # number of threads per block
  grid_size = cuda.gridDim.x    # number of blocks in the grid

  start = tx + ty * block_size
  end = nhills
  stride = block_size * grid_size
  
  for i in range(start, end, stride):
    doy = doy_init
    soil_temperature = 0
    if i < end:
      max_static = parameters[i,0]
      surface_factor= parameters[i,1] #0-1
      interflow_factor = parameters[i,2]#0-1
      tile_drain_factor = parameters[i,3]#0-1
      baseflow_factor = parameters[i,4]#0-1
      depth_td = parameters[i,5] #m
      surface_exponent = parameters[i,6]
      interflow_exponent = parameters[i,7]
      tile_drain_exponent = parameters[i,8]
      baseflow_exponent = parameters[i,9]
      temperature_limit= parameters[i,10]
      melt_factor = parameters[i,11]
      inf_factor = parameters[i,12]
      per_factor = parameters[i,13]
      for t in range(0,(timesteps)):
        doy+= 1/24
        
        _precipitation = precipitation[i,t] #mm/hour
        _evapotranspiration = evapotranspiration[i,int(doy-1)] #mm/day
        _temperature = temperature[i,int(doy-1)] #celsius
        soil_temperature = soil_temperature + 0.33*(_temperature - soil_temperature)
        x1=0 # input to static storage [m]
        snowmelt=0
        if _temperature>=temperature_limit:
          snowmelt = _temperature  * melt_factor * CF_MELTFACTOR * DT #m
          if snowmelt >= swe[i]:
            snowmelt = swe[i]
          swe[i] = swe[i] - snowmelt #m
          x1 = (CF_MMHR_M_MIN * DT * _precipitation) + snowmelt #m
        if _temperature<temperature_limit:
          val = CF_MMHR_M_MIN * DT * _precipitation #[m]
          swe[i] += val
          x1=0
        #x1 = (CF_MMHR_M_MIN * DT * _precipitation)#m
        
        x2=0 # water that goes to the surface
        x3=0 # water that goes to toplayer
        d1=0 #water that goes to static
        
        infiltration = ksat_toplayer[i]* inf_factor * CF_MMHR_M_MIN * DT #infiltration rate [m/min] to [m]
        percolation = ksat_bottomlayer[i]*per_factor * CF_MMHR_M_MIN * DT # [m/min] to [m]
        top_max = toplayer_porosity[i]* 15./100.
        if bedrock[i]<15/100:
          bedrock[i]=15/100.
        bottom_max = bottomlayer_porosity[i] * (bedrock[i]-15/100.)
        #tile cant be deeper than bedrock
        if depth_td > bedrock[i]:
          depth_td = bedrock[i]

          #print(top_max,bottom_max)
        #static storage
        if static[i]>0:
          #print(static[i])
          pass
          
        if static[i]> max_static:
          static[i] = max_static
        
        #x2 = x1 + static[i] - max_static
        #x2 = max(0,x2)
        if soil_temperature <=0:
        #if snowmelt>0:
          x2 = x1 #m
        else:
          x2 = x1 + static[i] - max_static
          x2 = max(0,x2)
        d1 = x1 - x2
        #out1 = _evapotranspiration * CF_MMDAY_M_MIN * DT #mm/day to m/min to m
        et = _evapotranspiration * CF_MMDAY_M_MIN * DT #mm/day to m/min to m
        out1 = min(et,static[i])
        et2 = et - out1
        static[i] += d1 - out1
        if static[i]<0:
          static[i]=0

        #surface storage
        
        if surface[i]>0:
          #print(surface[i],infiltration,x2)
          pass
        #cant infiltrate more than infiltration rate,
        # #cant infiltrate more than input x2
        # cant infiltrate more than available room in toplayer
        if(soil_temperature<=0):
          infiltration=0
        x3 = min(x2,infiltration,(top_max-toplayer[i])) 
        d2 = x2 - x3
        out2 = surface_factor * surface[i]**surface_exponent #[m]
        out2 = min(out2, surface[i])
        
        surface[i] += d2 - out2

        #toplayer storage
        #cant percolate more than percolation rate,
        # #cant percolate more than input x3
        # cant percolate more than available room in botomlayer
        x4 = min(x3,percolation,(bottom_max - bottomlayer[i]))
        d3 = x3 - x4
        #if d3>0 :
        #  print(t,d3,x3,x4,bottom_max,bottomlayer[i])
        if interflow_factor<=0:
          out3=0
        else:
          out3 = interflow_factor*toplayer[i] ** interflow_exponent  #[m]
          out3 = min(out3, toplayer[i])
          #et3 = et2
          #et2 = min(et2,toplayer[i])
          #et3 -=et2
          #out1+=et2
        toplayer[i] += d3 - out3 #- et2 #[m]

        #bottomlayer and tile
  

        d4 = x4
        bottomlayer[i]+= d4

        et3 = et2
        et2 = min(et2,bottomlayer[i])
        et3 -=et2
        out1+=et2
        
        #if et3>0:
        #    print(et,out1,et2,et3)
        bottomlayer[i]+= - et2
          
        if td_factor[i]>0.2:
          bottomlayer_below_td_max_storage = (bedrock[i] - depth_td)*bottomlayer_porosity[i]
          #if t==1:
          #  print(bottomlayer_below_td_max_storage,bottom_max)

          excess = bottomlayer[i]- bottomlayer_below_td_max_storage #negative if water table below td
          excess = max(excess,0)
          out4 = tile_drain_factor* excess **tile_drain_exponent 
          out5 = baseflow_factor * min(bottomlayer[i],bottomlayer_below_td_max_storage) ** baseflow_exponent
          bottomlayer[i]+=  - out4 - out5
        
        if td_factor[i]<=0.2:
          out4 = 0 #no td output
          out5 = bottomlayer[i] * baseflow_factor
          bottomlayer[i]+=  - out4 - out5
        
        bottomlayer[i] = max(0,bottomlayer[i])

        #aux variables
        STATE_RUNOFF=0
        model_output[STATE_RUNOFF,i,t] = (out2 + out3 + out4 + out5)* area_hillslope[i] / float(3600) #m3/s
        STATE_PRECIP=1
        model_output[STATE_PRECIP,i,t] = _precipitation * area_hillslope[i] # mm x m2
        STATE_ACTUALET=2
        model_output[STATE_ACTUALET,i,t] = CF_METER_TO_MM * out1 * area_hillslope[i] #[mm x m2]
        STATE_STATIC=3
        model_output[STATE_STATIC,i,t] = CF_METER_TO_MM * static[i] * area_hillslope[i] # [mm x m2]
        STATE_SURFACE=4
        model_output[STATE_SURFACE,i,t]=  CF_METER_TO_MM * surface[i] * area_hillslope[i] # mm x m2
        STATE_TOPLAYER=5
        #model_output[STATE_TOPLAYER,i,t]=  CF_METER_TO_MM * toplayer[i] * area_hillslope[i] # mm x m2
        model_output[STATE_TOPLAYER,i,t]=  toplayer[i] # m
        STATE_BOTTOMLAYER=6
        #model_output[STATE_BOTTOMLAYER,i,t]=  CF_METER_TO_MM * bottomlayer[i] * area_hillslope[i] # mm x m2
        model_output[STATE_BOTTOMLAYER,i,t]=  bottomlayer[i] # m
        STATE_SURFACE_FLOW=7
        model_output[STATE_SURFACE_FLOW,i,t]=  out2 * area_hillslope[i]/ float(3600) # [m3/s]
        STATE_INTERFLOW=8
        model_output[STATE_INTERFLOW,i,t]=  out3 * area_hillslope[i]/ float(3600) # [m3/s]
        STATE_TDFLOW=9
        model_output[STATE_TDFLOW,i,t]=  out4 * area_hillslope[i]/ float(3600) # [m3/s]
        STATE_BASEFLOW=10
        model_output[STATE_BASEFLOW,i,t]=  out5 * area_hillslope[i]/ float(3600) # [m3/s]


def run_cuda():

    N = 1000
    timesteps = 30*24
    timesteps = 8000
    doy = 1
    n_outputs = get_n_outputs()
    precipitation = cuda.to_device((np.random.uniform(0,1,size=(N,timesteps))))
    area_hillslope = cuda.to_device((np.random.uniform(1,100,size=N)))
    evapotranspiration= cuda.to_device((np.random.uniform(1,700,size=(N,timesteps))))
    temperature = cuda.to_device((np.random.uniform(0,10,size=(N,timesteps))))


    static=cuda.to_device(np.ones(N))
    swe=cuda.to_device(np.ones(N))
    surface=cuda.to_device(np.zeros(N))
    toplayer=cuda.to_device(np.ones(N))
    bottomlayer=cuda.to_device(np.ones(N))
    bedrock=cuda.to_device(np.ones(N)* float(1.5))  #m
    toplayer_max =cuda.to_device(np.ones(N)*float(0.5))#m
    bottomlayer_max =cuda.to_device(np.ones(N)*float(0.5))#m
    ksat_toplayer =cuda.to_device(np.ones(N)*float(8))#mm/h
    ksat_bottomlayer =cuda.to_device(np.ones(N)*float(6))#mm/h
    td_factor = cuda.to_device(np.ones(N))

    model_output = cuda.to_device(np.empty(shape=(n_outputs,N,timesteps),dtype=np.float32))

    max_static=0.15 #m
    surface_factor= 0.01 # adim, 0 to 1
    toplayer_factor = 0.01 #adim
    tile_drain_factor = 0.01
    bf_factor = 0.1
    depth_td = 1.4
    surface_exponent=2
    interflow_exponent=2
    tiledrain_exponent=2
    baseflow_exponent=2
    temperature_limit=0
    melt_factor=1
    inf_factor=1
    per_factor=1
    _par = [max_static,
      surface_factor,
      toplayer_factor,
      tile_drain_factor,
      bf_factor,
      depth_td,
      surface_exponent,
      interflow_exponent,
      tiledrain_exponent,
      baseflow_exponent,
      temperature_limit,
      melt_factor,inf_factor,per_factor]
    
    parameters = np.ones(shape=(N,len(_par)),dtype=np.float32)
    for i in range(len(_par)):
        parameters[:,i]=_par[i]
    parameters = cuda.to_device(parameters)
    #nvidia-settings -q CUDACores -t  .
    threads_per_block = 512
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    hlm_kernel[blocks_per_grid, threads_per_block](
      timesteps,N,doy,
      model_output,
      precipitation,
      evapotranspiration,
      temperature,
      area_hillslope,
      static,
      surface,
      toplayer,
      bottomlayer,
      swe,
      bedrock,
      toplayer_max,
      bottomlayer_max,
      ksat_toplayer,
      ksat_bottomlayer,
      td_factor,
      parameters)
    cuda.synchronize()
    print('done generation')

if __name__ == "__main__":
  run_cuda()
