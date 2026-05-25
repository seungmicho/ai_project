require('dotenv').config();

const express = require('express');
const axios = require('axios');

const app = express();
const PORT = 3000;

const KAKAO_REST_API_KEY = process.env.KAKAO_REST_API_KEY;
const TMAP_APP_KEY = process.env.TMAP_APP_KEY;

app.use(express.json());
app.use(express.static('public'));

function isValidTime(time) {
  return /^([01]\d|2[0-3]):([0-5]\d)$/.test(time);
}

function calculateDeparture(arrivalTime, travelTime, prepTime = 20, bufferTime = 10) {
  const [hour, minute] = arrivalTime.split(':').map(Number);

  let totalMinutes = hour * 60 + minute;
  totalMinutes -= (travelTime + prepTime + bufferTime);

  while (totalMinutes < 0) {
    totalMinutes += 24 * 60;
  }

  const depHour = Math.floor(totalMinutes / 60);
  const depMinute = totalMinutes % 60;

  return `${String(depHour).padStart(2, '0')}:${String(depMinute).padStart(2, '0')}`;
}

function getRemainMinutes(departureTime) {
  const nowMinutes = getNowMinutes();

  const [depHour, depMinute] = departureTime.split(':').map(Number);
  const departureMinutes = depHour * 60 + depMinute;

  return departureMinutes - nowMinutes;
}

function getRiskLevel(remain) {
  if (remain > 20) {
    return 'SAFE';
  } else if (remain > 5) {
    return 'CAUTION';
  } else if (remain >= 0) {
    return 'URGENT';
  } else {
    return 'LATE';
  }
}

function getStatusMessages(riskLevel, remain, currentArrivalTime) {
  switch (riskLevel) {
    case 'SAFE':
      return {
        mainMessage: '아직 여유가 있습니다.',
        detailMessage: `약 ${remain}분 후 출발하면 됩니다. 지금 출발하면 ${currentArrivalTime}에 도착합니다.`
      };

    case 'CAUTION':
      return {
        mainMessage: '곧 출발 준비를 하세요.',
        detailMessage: `현재 기준 ${remain}분 안에 출발해야 합니다. 지금 출발하면 ${currentArrivalTime}에 도착합니다.`
      };

    case 'URGENT':
      return {
        mainMessage: '지금 출발하세요.',
        detailMessage: `현재 기준 지금 출발해야 정시에 도착할 수 있습니다. 지금 출발하면 ${currentArrivalTime}에 도착합니다.`
      };

    case 'LATE':
      return {
        mainMessage: '이미 늦었습니다.',
        detailMessage: `지금 출발해도 약 ${Math.abs(remain)}분 늦을 예정입니다. 예상 도착 시각은 ${currentArrivalTime}입니다.`
      };

    default:
      return {
        mainMessage: '상태를 확인할 수 없습니다.',
        detailMessage: ''
      };
  }
}

function formatDuration(minutes) {
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;

  if (h > 0 && m > 0) return `${h}시간 ${m}분`;
  if (h > 0 && m === 0) return `${h}시간`;
  return `${m}분`;
}

function formatClockTime(totalMinutes) {
  let minutes = totalMinutes % (24 * 60);
  if (minutes < 0) minutes += 24 * 60;

  const hour = Math.floor(minutes / 60);
  const minute = minutes % 60;

  return `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
}

function getNowMinutes() {
  const now = new Date();
  return now.getHours() * 60 + now.getMinutes();
}

function getCurrentArrivalTime(travelTime) {
  const nowMinutes = getNowMinutes();
  return formatClockTime(nowMinutes + travelTime);
}

function getArrivalTimeFromDeparture(departureTime, travelTime) {
  const [hour, minute] = departureTime.split(':').map(Number);
  const departureMinutes = hour * 60 + minute;

  return formatClockTime(departureMinutes + travelTime);
}

function getSimpleSimulationStatus(remain) {
  if (remain > 20) return '여유';
  if (remain > 5) return '준비';
  if (remain >= 0) return '즉시 출발';
  return '지각';
}

// 장소명 -> 좌표
async function getCoordinates(keyword) {
  const url = 'https://dapi.kakao.com/v2/local/search/keyword.json';

  const response = await axios.get(url, {
    headers: {
      Authorization: `KakaoAK ${KAKAO_REST_API_KEY}`
    },
    params: {
      query: keyword
    }
  });

  const documents = response.data.documents;

  if (!documents || documents.length === 0) {
    return null;
  }

  return {
    x: Number(documents[0].x),
    y: Number(documents[0].y),
    placeName: documents[0].place_name,
    addressName: documents[0].address_name
  };
}

// 자동차 경로 + 시간 + 경로 좌표
async function getCarRoute(origin, destination) {
  const url = 'https://apis-navi.kakaomobility.com/v1/directions';

  const response = await axios.get(url, {
    headers: {
      Authorization: `KakaoAK ${KAKAO_REST_API_KEY}`
    },
    params: {
      origin: `${origin.x},${origin.y}`,
      destination: `${destination.x},${destination.y}`
    }
  });

  const route = response.data?.routes?.[0];
  const summary = route?.summary;
  const sections = route?.sections;

  if (!summary || !sections || sections.length === 0) {
    throw new Error('자동차 경로를 찾지 못했습니다.');
  }

  const durationSec = summary.duration;
  const distanceMeter = summary.distance;

  const path = [];

  for (const section of sections) {
    for (const road of section.roads) {
      const vertexes = road.vertexes;
      for (let i = 0; i < vertexes.length; i += 2) {
        path.push({
          lng: vertexes[i],
          lat: vertexes[i + 1]
        });
      }
    }
  }

  return {
    transportLabel: '자동차',
    travelTime: Math.ceil(durationSec / 60),
    distanceKm: (distanceMeter / 1000).toFixed(1),
    path,
    routeSteps: []
  };
}

function parseLineString(lineString) {
  if (!lineString || typeof lineString !== 'string') {
    return [];
  }

  return lineString
    .trim()
    .split(/\s+/)
    .map(pair => {
      const [lng, lat] = pair.split(',').map(Number);

      if (Number.isNaN(lng) || Number.isNaN(lat)) {
        return null;
      }

      return { lng, lat };
    })
    .filter(Boolean);
}

function extractTransitPath(itinerary) {
  const path = [];

  for (const leg of itinerary.legs || []) {
    // 도보 구간
    if (leg.mode === 'WALK') {
      for (const step of leg.steps || []) {
        const stepPath = parseLineString(step.linestring);
        path.push(...stepPath);
      }
    }

    // 버스/지하철 구간
    if (leg.mode === 'BUS' || leg.mode === 'SUBWAY') {
      const passShapePath = parseLineString(leg.passShape?.linestring);
      path.push(...passShapePath);
    }
  }

  return path;
}

function extractTransitSegments(itinerary) {
  return (itinerary.legs || []).map((leg) => {
    let segmentPath = [];

    if (leg.mode === 'WALK') {
      if (leg.steps && leg.steps.length > 0) {
        for (const step of leg.steps) {
          segmentPath.push(...parseLineString(step.linestring));
        }
      } else {
        segmentPath.push(...parseLineString(leg.passShape?.linestring));
      }
    }

    if (leg.mode === 'BUS' || leg.mode === 'SUBWAY') {
      segmentPath.push(...parseLineString(leg.passShape?.linestring));
    }

    return {
      mode: leg.mode ?? '',
      sectionTime: leg.sectionTime ?? null,
      distance: leg.distance ?? null,
      startName: leg.start?.name ?? '',
      endName: leg.end?.name ?? '',
      start: leg.start
        ? { lng: leg.start.lon, lat: leg.start.lat }
        : null,
      end: leg.end
        ? { lng: leg.end.lon, lat: leg.end.lat }
        : null,
      routeName: leg.route ?? leg.Lane?.[0]?.route ?? '',
      routeColor: leg.routeColor ?? null,
      descriptions: (leg.steps || []).map(step => step.description).filter(Boolean),
      path: segmentPath
    };
  }).filter(segment => segment.path.length > 0);
}

// 대중교통 경로 + 시간
async function getTransitRoute(origin, destination) {
  const url = 'https://apis.openapi.sk.com/transit/routes';

  const response = await axios.post(
    url,
    {
      startX: String(origin.x),
      startY: String(origin.y),
      endX: String(destination.x),
      endY: String(destination.y),
      count: 1,
      lang: 0,
      format: 'json'
    },
    {
      headers: {
        appKey: TMAP_APP_KEY,
        Accept: 'application/json',
        'Content-Type': 'application/json'
      }
    }
  );

  const itinerary = response.data?.metaData?.plan?.itineraries?.[0];

  console.log(
    'Transit legs:',
    JSON.stringify(itinerary.legs, null, 2)
  );

  if (!itinerary) {
    throw new Error('대중교통 경로를 찾지 못했습니다.');
  }

  const totalTimeSec = itinerary.totalTime;
  const totalDistanceMeter = itinerary.totalDistance;

  const routeSteps = (itinerary.legs || []).map((leg) => {
    return {
      mode: leg.mode ?? '',
      sectionTime: leg.sectionTime ?? null,
      distance: leg.distance ?? null,
      startName: leg.start?.name ?? '',
      endName: leg.end?.name ?? '',
      routeName: leg.route ?? leg.routeName ?? leg.lane?.[0]?.name ?? ''
    };
  });

  const path = extractTransitPath(itinerary);
  const routeSegments = extractTransitSegments(itinerary);

  return {
    transportLabel: '대중교통',
    travelTime: Math.ceil(totalTimeSec / 60),
    distanceKm: totalDistanceMeter
      ? (totalDistanceMeter / 1000).toFixed(1)
      : null,
    transferCount: itinerary.transferCount ?? null,
    fare: itinerary.fare?.regular?.totalFare ?? null,
    walkDistance: itinerary.totalWalkDistance ?? null,
    path,
    routeSteps,
    routeSegments
  };
}

async function getRouteByMode(transport, origin, destination) {
  if (transport === 'transit') {
    return await getTransitRoute(origin, destination);
  }

  return await getCarRoute(origin, destination);
}

app.post('/calculate', async (req, res) => {
  const { start, end, time, transport = 'car' } = req.body;

  if (!start || !end || !time) {
    return res.json({ error: '필수 입력값이 비어 있습니다.' });
  }

  if (!isValidTime(time)) {
    return res.json({ error: '도착 시간 형식이 올바르지 않습니다. 예: 09:30' });
  }

  try {
    const origin = await getCoordinates(start);
    const destination = await getCoordinates(end);

    if (!origin || !destination) {
      return res.json({ error: '출발지 또는 도착지를 찾지 못했습니다.' });
    }

    const routeInfo = await getRouteByMode(transport, origin, destination);
    const departure = calculateDeparture(time, routeInfo.travelTime);
    const currentArrivalTime = getCurrentArrivalTime(routeInfo.travelTime);
    const remain = getRemainMinutes(departure);
    const riskLevel = getRiskLevel(remain);
    const { mainMessage, detailMessage } = getStatusMessages(riskLevel, remain, currentArrivalTime);

    const recommendedArrivalTime = getArrivalTimeFromDeparture(
      departure,
      routeInfo.travelTime
    );

    const delayedDepartureMinutes =
      (() => {
        const [h, m] = departure.split(':').map(Number);
        return h * 60 + m + 10;
      })();

    const delayedDepartureTime = formatClockTime(delayedDepartureMinutes);
    const delayedArrivalTime = formatClockTime(delayedDepartureMinutes + routeInfo.travelTime);
    const delayedRemain = getRemainMinutes(delayedDepartureTime);

    const simulations = [
      {
        label: '지금 출발',
        departureTime: formatClockTime(getNowMinutes()),
        arrivalTime: currentArrivalTime,
        remain: remain,
        status: remain >= 0 ? '가능' : `약 ${Math.abs(remain)}분 늦음`
      },
      {
        label: '추천 출발',
        departureTime: departure,
        arrivalTime: recommendedArrivalTime,
        remain: 0,
        status: '정시 도착'
      },
      {
        label: '10분 늦게 출발',
        departureTime: delayedDepartureTime,
        arrivalTime: delayedArrivalTime,
        remain: delayedRemain,
        status: delayedRemain >= 0 ? '가능' : `약 ${Math.abs(delayedRemain)}분 늦음`
      }
    ];

    return res.json({
      transportLabel: routeInfo.transportLabel || '자동차',
      travelTime: routeInfo.travelTime,
      travelTimeText: formatDuration(routeInfo.travelTime),
      distanceKm: routeInfo.distanceKm,
      transferCount: routeInfo.transferCount ?? null,
      fare: routeInfo.fare ?? null,
      walkDistance: routeInfo.walkDistance ?? null,
      departure,
      currentArrivalTime,
      remain,
      riskLevel,
      mainMessage,
      detailMessage,
      simulations,
      startPlaceName: origin.placeName,
      startAddressName: origin.addressName,
      endPlaceName: destination.placeName,
      endAddressName: destination.addressName,
      origin,
      destination,
      path: routeInfo.path,
      routeSteps: routeInfo.routeSteps ?? [],
      routeSegments: routeInfo.routeSegments ?? []
    });
  } catch (error) {
    console.error('Server error:', error.response?.data || error.message);
    return res.json({
      error: '서버 처리 중 오류가 발생했습니다.'
    });
  }
});

app.listen(PORT, () => {
  console.log(`서버 실행: http://localhost:${PORT}`);
});


