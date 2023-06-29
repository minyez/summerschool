FC = ftn
CXX = CC

FSRCSUFFIX = F90
CXXSRCSUFFIX = cpp
FOBJSUFFIX = f
CXXOBJSUFFIX = C

F_SRCS = $(wildcard *.$(FSRCSUFFIX))
CPP_SRCS = $(wildcard *.$(CXXSRCSUFFIX))

F_OBJS = $(patsubst %.$(FSRCSUFFIX),%_f,$(F_SRCS))
CPP_OBJS = $(patsubst %.$(CXXSRCSUFFIX),%_C,$(CPP_SRCS))

all: $(F_OBJS) $(CPP_OBJS)

%_$(FOBJSUFFIX): %.$(FSRCSUFFIX)
	$(FC) $< -o $@

%_$(CXXOBJSUFFIX): %.$(CXXSRCSUFFIX)
	$(CXX) $< -o $@

clean:
	rm -f *_$(CXXOBJSUFFIX) *_$(FOBJSUFFIX)
