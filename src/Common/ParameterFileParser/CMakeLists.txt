# Sources of non-templated classes.
SET(param_SRCS
lex.yy.cxx
y.tab.cxx
param.cxx
)

ADD_LIBRARY(param ${param_SRCS})

TARGET_LINK_LIBRARIES(param ITKCommon)
