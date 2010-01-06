#ifndef __itkParabolicUtils_h
#define __itkParabolicUtils_h

#include <itkArray.h>

#include "itkProgressReporter.h"
namespace itk {
template <class LineBufferType, class RealType, bool doDilate>
void DoLine(LineBufferType &LineBuf, LineBufferType &tmpLineBuf, 
	    const RealType magnitude, const RealType m_Extreme)
{
  // contact point algorithm
  long koffset = 0, newcontact=0;  // how far away the search starts.

  const long LineLength = LineBuf.size();
  // negative half of the parabola
  for (long pos = 0; pos < LineLength; pos++)
    {
    RealType BaseVal = (RealType)m_Extreme; // the base value for
					    // comparison
    for (long krange = koffset; krange <= 0; krange++)
      {
      // difference needs to be paramaterised
      RealType T = LineBuf[pos + krange] - magnitude * krange * krange;
      // switch on template parameter - hopefully gets optimized away.
      if (doDilate ? (T >= BaseVal) : (T <= BaseVal) )
	{
	BaseVal = T;
	newcontact = krange;
	}
      }
    tmpLineBuf[pos] = BaseVal;
    koffset = newcontact - 1;
    }
  // positive half of parabola
  koffset = newcontact = 0;
  for (long pos = LineLength - 1; pos >= 0; pos--)
    {
    RealType BaseVal = (RealType)m_Extreme; // the base value for comparison
    for (long krange = koffset; krange >= 0; krange--)
      {
      RealType T = tmpLineBuf[pos + krange] - magnitude * krange * krange;
      if (doDilate ? (T >= BaseVal) : (T <= BaseVal))
	{
	BaseVal = T;
	newcontact = krange;
	}
      }
    LineBuf[pos] = BaseVal;
    koffset = newcontact + 1;
    }
}

template <class TInIter, class TOutIter, class RealType, 
	  class OutputPixelType, bool doDilate>
void doOneDimension(TInIter &inputIterator, TOutIter &outputIterator, 
		    ProgressReporter &progress,
		    const long LineLength, 
		    const unsigned direction,
		    const int m_MagnitudeSign,
		    const bool m_UseImageSpacing,
		    const RealType m_Extreme,
		    const RealType image_scale,
		    const RealType Sigma)
{
//  typedef typename std::vector<RealType> LineBufferType;

  // message from M.Starring suggested performance gain using Array
  // instead of std::vector.
  typedef typename itk::Array<RealType> LineBufferType;
  RealType iscale = 1.0;
  if (m_UseImageSpacing)
    {
    iscale = image_scale;
    }
  const RealType magnitude = m_MagnitudeSign * 1.0/(2.0 * Sigma/(iscale*iscale));
  LineBufferType LineBuf(LineLength);
  LineBufferType tmpLineBuf(LineLength);
  inputIterator.SetDirection(direction);
  outputIterator.SetDirection(direction);
  inputIterator.GoToBegin();
  outputIterator.GoToBegin();
  
  while( !inputIterator.IsAtEnd() && !outputIterator.IsAtEnd() )
    {
    // process this direction
    // fetch the line into the buffer - this methodology is like
    // the gaussian filters
    unsigned int i=0;
    while( !inputIterator.IsAtEndOfLine() )
      {
      LineBuf[i++]      = static_cast<RealType>(inputIterator.Get());
      ++inputIterator;
      }
    
    DoLine<LineBufferType, RealType, doDilate>(LineBuf, tmpLineBuf, magnitude, m_Extreme);
    // copy the line back
    unsigned int j=0;
    while( !outputIterator.IsAtEndOfLine() )
      {
      outputIterator.Set( static_cast<OutputPixelType>( LineBuf[j++] ) );
      ++outputIterator;
      }
    
    // now onto the next line
    inputIterator.NextLine();
    outputIterator.NextLine();
    progress.CompletedPixel();
    }
}


}
#endif
