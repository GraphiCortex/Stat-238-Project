create table cell (
 id integer,       -- Id used to match original row number in MatLab PyrIntMap.Map matrix
 topdir string,    -- top level directory containing data
 animal string,    -- name of animal
 ele integer,      -- electrode number
 clu integer,      -- ID # in cluster files
 region string,    -- brain region
 nexciting integer,   -- number of cells this cell monosynaptically excited 
 ninhibiting integer, -- number of cells this cell monosynaptically inhibited 
 exciting integer,    -- physiologically identified exciting cells based on CCG analysis
 inhibiting integer,  -- physiologically identified inhibiting cells based on CCG analysis
      -- (Detailed method can be found in Mizuseki  Sirota Pastalkova and Buzsaki., 2009 Neuron paper.)
 excited integer,     -- based on cross-correlogram analysis, the cell is monosynaptically excited by other cells
 inhibited integer,   -- based on cross-correlogram analysis, the cell is monosynaptically inhibited by other cells
 fireRate real,       -- meanISI = mean(bootstrp(100,'mean',ISI)); fireRate = SampleRate/MeanISI; ISI is interspike intervals.
 totalFireRate real,  -- num of spikes divided by total recording length
 cellType string      -- 'p'=pyramidal, 'i'=interneuron, 'n'=not identified as pyramidal or interneuron
);


create table session (
  id integer,     -- matches row in original MatLab Beh matrix
  topdir string,  -- directory in data set containing data (tar.gz) files
  session string, -- individual session name (corresponds to name of tar.gz file having data)
  behavior string, -- behavior, one of: Mwheel, Open, Tmaze, Zigzag, bigSquare, bigSquarePlus, circle
                   -- linear, linearOne, linearTwo, midSquare, plus, sleep, wheel, wheel_home
  familiarity integer, -- number of times animal has done task, 1=animal did task for first time, 
                       -- 2=second time, 3=third time, 10=10 or more
  duration real   -- recording length in seconds
);


create table file (
  -- information about files in hc3 dataset
  topdir string,  -- directory in data set containg data (tar.gz) files
  session string, -- individual session name (corresponds to name of tar.gz file having data)
  size integer,   -- number of bytes in tar.gz file
  video_type string, -- 'mpg', 'm1v' or '-' (for no video file)
  video_size integer -- size of video file, or 0 if no video file
);

create table epos (
  -- has electrode positions for each top level directory
  -- Note, some regions do not match that in cell table.
  -- Those that differ have following meanings:
  --   DGCA3: not sure sure if the electrode is DG or CA3.
  --   Ctx: somewhere in the cortex (above the hippocampus)
  --   CA: somewhere in the hippocampus (do not know if it is CA1, CA3 or DG)
  topdir string,  -- directory in data set containing data (tar.gz) file
  animal string,  -- animal name
  e1 string,      -- region for electrode 1 
  e2 string,      -- region for electrode 2
  e3 string,      -- region for electrode 3
  e4 string,      -- region for electrode 4
  e5 string,      -- region for electrode 5
  e6 string,      -- region for electrode 6
  e7 string,      -- region for electrode 7
  e8 string,      -- region for electrode 8
  e9 string,      -- region for electrode 9
  e10 string,     -- region for electrode 10
  e11 string,     -- region for electrode 11
  e12 string,     -- region for electrode 12
  e13 string,     -- region for electrode 13
  e14 string,     -- region for electrode 14
  e15 string,     -- region for electrode 15
  e16 string      -- region for electrode 16
);

